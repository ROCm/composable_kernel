# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host data of verified search calls, computed once per problem and run.

A verified search call runs one kernel in a fresh worker process, which would
otherwise regenerate the problem's random inputs and recompute its fp32
reference on every call. A search run keeps them in a temp directory of its
own (``HostCache``, in the driver) that its workers read from, or write to on
a miss (``cached``, in the worker). The search fills a problem's entry before
its first measure call, in a worker that only does that (``prepare``):

- an entry is a directory ``<dir>/<key>`` of ``.npy`` files and a
  ``meta.json``, written under a temp name and renamed into place, so a reader
  sees a complete entry or none;
- ``key`` hashes the fields the op says decide the data (problem, dtype,
  layout, input seed, ...); ``meta.json`` repeats them, and a mismatch is a
  miss;
- a key has one writer per run: under the directory's lock, a worker that
  missed claims the key and reserves the bytes of its entry, or does not
  write it. Workers that miss while another writes the key use what they
  computed, as without the cache;
- an entry larger than ``max_entry_bytes`` is not written, nor one whose
  reservation would take the run's reservations past ``max_total_bytes`` or
  leave the filesystem less than ``MIN_FREE_FRACTION`` free once every
  reserved entry is written. The caps hold with any number of workers;
- a reader maps an entry's arrays read-only (``np.load`` with
  ``mmap_mode="r"``) instead of copying them: on a tmpfs its pages are the
  entry's own, and on a disk they are page cache, so workers reading one
  entry at once add no copy of it each;
- an error opening or writing an entry is a miss: the cache never changes a
  result, only how often it is computed. A writer that fails gives its claim
  back; one that is killed keeps it, and its key is computed on every call
  for the rest of the run. A mapped entry's pages are read when the worker
  uses them: an I/O error then (a failing disk, never a tmpfs) ends that
  worker with SIGBUS, and its call is a run_error;
- ``room`` is what the cache may still store, for a caller that must keep
  memory for it (a tmpfs holds its entries in RAM), and ``drop`` removes the
  entries no call will read.

``HostCache.close`` (also on interpreter exit) removes the directory. Importing
this module does not import numpy. Writing needs ``fcntl`` (POSIX); without
it nothing is stored. The directory belongs on a local filesystem (a tmpfs or
a local disk): on NFS a worker that waits for the lock waits out the client's
lock retry backoff, seconds to minutes inside its call's timeout.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

FORMAT = 1  # part of every key: a change of the entry layout misses old entries
MIN_FREE_FRACTION = 0.1  # of the filesystem, left free by every write
GIB = 1 << 30
PREFIX = "ck_search_host_cache-"
CLAIM = ".w-"  # ".w-<key>-<bytes reserved>": a key's writer, kept once written
NPY_HEADER_BYTES = 4096  # reserved per .npy file (its header is 128 bytes here)
NFS_TYPES = ("nfs", "nfs4")  # filesystems where waiting for the lock backs off


def entry_key(fields: Mapping) -> str:
    """The entry name of ``fields`` (JSON data)."""
    text = json.dumps({"format": FORMAT, "fields": fields}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def usage(root) -> Tuple[int, int]:
    """(entries, bytes) under ``root``; temp entries count bytes only."""
    entries = size = 0
    for path, _dirs, files in os.walk(root):
        if "meta.json" in files and not Path(path).name.startswith(".tmp-"):
            entries += 1
        for f in files:
            try:
                size += os.stat(os.path.join(path, f)).st_size
            except OSError:
                pass
    return entries, size


def fs_type(path, mounts="/proc/self/mounts") -> str:
    """The type of the filesystem holding ``path``, from the Linux mount table
    ``mounts`` (the last mount on the longest matching mount point); "" when
    unknown."""
    path, found, kind = os.path.realpath(path), "", ""
    try:
        with open(mounts) as f:
            for line in f:
                mnt, typ = line.split()[1:3]
                mnt = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), mnt)
                if ((path + "/").startswith(mnt.rstrip("/") + "/")
                        and len(mnt) >= len(found)):
                    found, kind = mnt, typ
    except (OSError, ValueError):
        return ""
    return kind


class HostCache:
    """A search run's cache directory: a new temp directory under ``parent``
    (the system temp directory when None). ``spec`` is what a worker gets.
    ``max_total_bytes`` None means a quarter of the free space of that
    filesystem now."""

    def __init__(self, parent=None, max_entry_bytes: int = 4 * GIB,
                 max_total_bytes: Optional[int] = None):
        self.dir = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent))
        if max_total_bytes is None:
            max_total_bytes = shutil.disk_usage(self.dir).free // 4
        self.spec = {"dir": str(self.dir), "max_entry_bytes": int(max_entry_bytes),
                     "max_total_bytes": int(max_total_bytes)}
        atexit.register(self.close)

    def close(self) -> Tuple[int, int]:
        """Remove the directory; returns its (entries, bytes) before."""
        atexit.unregister(self.close)
        found = usage(self.dir)
        shutil.rmtree(self.dir, ignore_errors=True)
        return found

    def __enter__(self) -> "HostCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextlib.contextmanager
def run_cache(settings, log=print):
    """A ``HostCache`` for the run's workers, from ``settings`` (the search's
    ``HostCacheSettings``), removed on exit with one ``log`` line; None when
    ``max_total_gib`` is 0. A filesystem already under ``MIN_FREE_FRACTION``
    free can store no entry: the run then starts with a warning line. On NFS
    it starts with a note to use a local filesystem. A cache that cannot be
    created (``dir`` read-only, say) is None, with a warning line: the cache
    is optional."""
    if settings.max_total_gib == 0:
        yield None
        return
    total = settings.max_total_gib
    try:
        cache = HostCache(settings.dir, int(settings.max_entry_gib * GIB),
                          None if total is None else int(total * GIB))
    except OSError as e:
        log(f"  Host cache: WARNING: cannot create it under "
            f"{settings.dir or tempfile.gettempdir()} ({e.strerror or e}); the run "
            "goes on without it")
        yield None
        return
    try:
        disk = shutil.disk_usage(cache.dir)
        if disk.free <= MIN_FREE_FRACTION * disk.total:
            log(f"  Host cache: WARNING: stores nothing: {disk.free / GIB:.1f} of "
                f"{disk.total / GIB:.1f} GiB free under {cache.dir.parent}, and a write "
                f"must leave {MIN_FREE_FRACTION:.0%} free; set host_cache.dir to a tmpfs "
                "or another local filesystem")
        kind = fs_type(cache.dir)
        if kind in NFS_TYPES:
            log(f"  Host cache: NOTE: {cache.dir.parent} is on {kind}, where workers that "
                "miss at once can wait seconds to minutes of their call's timeout for the "
                "cache's lock; set host_cache.dir to a local filesystem (a tmpfs or a "
                "local disk)")
        yield cache
    finally:
        entries, size = cache.close()
        log(f"  Host cache: {entries} entries, {size / GIB:.2f} GiB, removed ({cache.dir})")


def cached(spec: Optional[Mapping], fields: Mapping,
           compute: Callable[[], Dict[str, object]]) -> Dict[str, object]:
    """``compute()``'s named numpy arrays, mapped read-only from the entry of
    ``fields`` in the cache ``spec`` (a ``HostCache.spec``) when it has one,
    else computed and, within the caps, published. ``spec`` None:
    ``compute()``."""
    if not spec:
        return compute()
    import numpy as np

    fields, path = _entry(spec, fields)
    try:
        meta = json.loads((path / "meta.json").read_text())
        if meta["fields"] == fields:
            return {n: np.load(path / f"{n}.npy", mmap_mode="r", allow_pickle=False)
                    for n in meta["arrays"]}
    except Exception:  # noqa: BLE001 - no entry, or an unreadable one: compute
        pass
    data = compute()
    try:
        _publish(spec, path, fields, data)
    except Exception:  # noqa: BLE001 - the result does not need the entry
        pass
    return data


def prepare(spec: Optional[Mapping], fields: Mapping,
            compute: Callable[[], Dict[str, object]]) -> bool:
    """Fill the entry of ``fields`` ahead of the calls that read it (as
    ``cached`` does on a miss, unless the cache holds it already). Returns
    whether the cache holds it now: not when ``spec`` is None, the caps
    refuse it or writing it failed."""
    if not stored(spec, fields):
        cached(spec, fields, compute)
    return stored(spec, fields)


def stored(spec: Optional[Mapping], fields: Mapping) -> bool:
    """Whether the cache ``spec`` holds a complete entry of ``fields``."""
    if not spec:
        return False
    fields, path = _entry(spec, fields)
    try:
        meta = json.loads((path / "meta.json").read_text())
        return meta["fields"] == fields and all(
            (path / f"{n}.npy").is_file() for n in meta["arrays"])
    except Exception:  # noqa: BLE001 - no entry, or an unreadable one
        return False


def room(spec: Optional[Mapping]) -> int:
    """Bytes the cache ``spec`` may still store: ``max_total_bytes`` less its
    reservations, and no more than its filesystem takes while leaving
    ``MIN_FREE_FRACTION`` free once every reserved entry is written (as
    ``_claim`` decides); 0 without a cache or with one it cannot read."""
    if not spec:
        return 0
    root = Path(spec["dir"])
    try:
        found = claims(root)
        pending = sum(n for k, n in found.items() if not (root / k).is_dir())
        disk = shutil.disk_usage(root)
    except OSError:
        return 0
    return max(0, min(spec["max_total_bytes"] - sum(found.values()),
                      int(disk.free - pending - MIN_FREE_FRACTION * disk.total)))


def drop(spec: Optional[Mapping], match: Callable[[Mapping], bool]) -> int:
    """Remove the entries of the cache ``spec`` whose fields ``match``, and
    their writers' claims (a later call may write them again); returns how
    many were removed."""
    if not spec:
        return 0
    root, removed = Path(spec["dir"]), 0
    with contextlib.suppress(OSError):
        for key, size in claims(root).items():
            try:
                if not match(json.loads((root / key / "meta.json").read_text())["fields"]):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            shutil.rmtree(root / key, ignore_errors=True)
            with contextlib.suppress(OSError):
                (root / f"{CLAIM}{key}-{size}").unlink()
            removed += 1
    return removed


def _entry(spec: Mapping, fields: Mapping) -> Tuple[object, Path]:
    """``fields`` as meta.json reads them back, and their entry's path."""
    fields = json.loads(json.dumps(fields))
    return fields, Path(spec["dir"]) / entry_key(fields)


def claims(root) -> Dict[str, int]:
    """{key: bytes reserved} of every writer that claimed a key under ``root``."""
    found = {}
    for name in os.listdir(root):
        if name.startswith(CLAIM):
            key, size = name[len(CLAIM):].rsplit("-", 1)
            found[key] = int(size)
    return found


def _claim(root: Path, key: str, size: int, max_total: int) -> Optional[Path]:
    """Under the lock of ``root``: claim ``key`` with ``size`` bytes reserved,
    unless it has a writer already or the caps refuse it."""
    import fcntl

    with open(root / ".lock", "a") as lock:  # fails once the run removed root
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
        found = claims(root)
        if key in found:
            return None
        # Bytes reserved but not on disk yet: entries still being written.
        pending = sum(n for k, n in found.items() if not (root / k).is_dir())
        disk = shutil.disk_usage(root)
        if (sum(found.values()) + size > max_total
                or disk.free - pending - size < MIN_FREE_FRACTION * disk.total):
            return None
        claim = root / f"{CLAIM}{key}-{size}"
        claim.touch()
        return claim


def _publish(spec: Mapping, path: Path, fields, data: Mapping) -> None:
    import numpy as np

    if sum(a.nbytes for a in data.values()) > spec["max_entry_bytes"]:
        return
    meta = json.dumps({"fields": fields, "arrays": sorted(data)})
    size = len(meta) + sum(a.nbytes + NPY_HEADER_BYTES for a in data.values())
    root = path.parent
    claim = _claim(root, path.name, size, spec["max_total_bytes"])
    if claim is None:
        return
    tmp = root / f".tmp-{path.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    written = False
    try:
        os.mkdir(tmp)
        for name, a in data.items():
            np.save(tmp / f"{name}.npy", a, allow_pickle=False)
        (tmp / "meta.json").write_text(meta)
        os.rename(tmp, path)  # atomic
        written = True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # gone after the rename
        if not written:
            with contextlib.suppress(OSError):
                claim.unlink()  # a later call may write the key

#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Rejects paths that are long enough to break a Windows build.
#
# Windows resolves paths against a classic MAX_PATH of 260 characters. Some
# build tools -- notably ninja's Stat() when it re-generates build.ninja -- do
# not honour the LongPathsEnabled registry setting, so a source file whose
# absolute path exceeds 260 characters fails the build outright:
#
#   ninja: error: rebuilding 'build.ninja': Stat(C:\...\device_grouped_conv2d_
#   fwd_bias_bn_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instance.in):
#   Filename longer than 260 characters
#
# The absolute path is the checkout prefix plus the repository-relative path.
# Only the second half is ours to control, so that is what this check bounds:
#
#   limit = MAX_PATH(260) - checkout prefix budget(60) = 200 characters
#
# The 60-character budget is the prefix used by the Windows nightly CI that
# first hit this (C:\actions-runner\_work\rocm-npi-dev\therock\rocm-libraries\),
# so a path that passes this check builds on Windows anywhere that prefix fits.
#
# A second, independent limit rejects individual file names over NAME_MAX(255),
# which no common filesystem (ext4, NTFS, APFS) can store at all.
#
# Used both by the Jenkinsfile "Determine CI Execution" stage -- where
# check_changed_path_length.sh feeds it the files a pull request adds or
# renames, so pre-existing long paths do not block unrelated work -- and by
# developers locally:
#
#   git ls-files | xargs script/check_path_length.sh   # audit the whole tree
#
# Paths may be given relative to the repository root, relative to the current
# directory, or absolute; all are normalised to repository-relative before
# measuring, with "." and ".." components resolved.
#
# Usage: ./check_path_length.sh <file1> <file2> ...
#
# Overrides (all optional):
#   CK_MAX_PATH_TOTAL   total path budget                      (default 260)
#   CK_PREFIX_BUDGET    characters reserved for the checkout   (default 60)
#   CK_MAX_PATH_LEN     repo-relative limit, overrides the two above
#   CK_MAX_NAME_LEN     single file name limit                 (default 255)

max_total=${CK_MAX_PATH_TOTAL:-260}
prefix_budget=${CK_PREFIX_BUDGET:-60}
max_path=${CK_MAX_PATH_LEN:-$((max_total - prefix_budget))}
max_name=${CK_MAX_NAME_LEN:-255}

# Where we are inside the repository, so a path given relative to the current
# directory can be reported (and measured) as the repository-relative path that
# Windows will actually resolve. Both are empty outside a git work tree, in
# which case arguments are measured exactly as given.
repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || repo_root=""
repo_prefix=$(git rev-parse --show-prefix 2>/dev/null) || repo_prefix=""

# Lexically normalise an absolute path: collapse repeated slashes, drop "."
# and resolve ".." textually. Textually, not via the filesystem, so a symlink
# is not dereferenced and a path that does not exist still normalises. Used
# only when GNU realpath is unavailable.
lexical_abs() {
    local part out=()
    local IFS=/
    for part in $1; do
        case $part in
            '' | .) ;;
            ..) (( ${#out[@]} )) && unset 'out[${#out[@]}-1]' ;;
            *)  out+=("$part") ;;
        esac
    done
    printf '/%s' "${out[*]}"
}

# Map an absolute path to its repository-relative form, resolving "." and ".."
# without dereferencing symlinks. Returns 1 if the path lies outside the
# repository, which has no repository-relative form and so cannot be measured.
repo_relative() {
    local abs=$1 out
    # -m: do not require the path to exist. -s: do not resolve symlinks, so a
    # symlink is measured at the path git records rather than at its target.
    if ! out=$(realpath -m -s --relative-to="$repo_root" -- "$abs" 2>/dev/null); then
        abs=$(lexical_abs "$abs")
        case $abs in
            "$repo_root")   out=. ;;
            "$repo_root"/*) out=${abs#"$repo_root"/} ;;
            *)              return 1 ;;
        esac
    fi
    [[ $out != .. && $out != ../* ]] || return 1
    printf '%s' "$out"
}

exit_code=0

for file in "$@"; do
    file=${file#./}
    [[ -n "$file" ]] || continue

    # Prefer the current-directory reading (matches shell semantics); fall back
    # to treating the argument as already repository-relative.
    #
    # -L as well as -e: -e dereferences, so it is false for a symlink whose
    # target does not exist. Git tracks such an entry (mode 120000) like any
    # other, and Windows still has to resolve its path, so a dangling symlink
    # must be measured rather than skipped.
    #
    # The argument is normalised rather than merely prefixed: "../f.cpp" from a
    # subdirectory is the repository-relative "f.cpp", not "sub/../f.cpp", and
    # an absolute path inside the repository is measured at its repository-
    # relative length. Prefixing without normalising over-counts, which can
    # reject a path that is in fact within the limit.
    if [[ -z "$repo_root" ]]; then
        # Outside a git work tree there is nothing to be relative to, so the
        # argument is measured exactly as given.
        [[ -e "$file" || -L "$file" ]] || continue
        path="$file"
    elif [[ -e "$file" || -L "$file" ]]; then
        # Relative to the current directory, which git reports as repo_prefix.
        if [[ $file == /* ]]; then
            abs="$file"
        else
            abs="${repo_root}/${repo_prefix}${file}"
        fi
        path=$(repo_relative "$abs") || continue
    elif [[ -e "${repo_root}/${file}" || -L "${repo_root}/${file}" ]]; then
        path=$(repo_relative "${repo_root}/${file}") || continue
    else
        continue
    fi

    if (( ${#path} > max_path )); then
        echo "ERROR: $path"
        echo "  repository-relative path is ${#path} characters, limit is ${max_path}" \
             "(MAX_PATH ${max_total} minus a ${prefix_budget}-character checkout prefix)"
        echo "  Fix: shorten a directory or file name by at least" \
             "$(( ${#path} - max_path )) character(s); this path breaks Windows builds."
        exit_code=1
    fi

    name=${path##*/}
    if (( ${#name} > max_name )); then
        echo "ERROR: $path"
        echo "  file name is ${#name} characters, limit is ${max_name}"
        echo "  Fix: shorten the file name by at least $(( ${#name} - max_name )) character(s);" \
             "no common filesystem can store it."
        exit_code=1
    fi
done

exit $exit_code

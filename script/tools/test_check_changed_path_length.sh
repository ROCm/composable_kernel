#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Unit test (pure bash) for script/check_changed_path_length.sh.
#
# The wrapper's job is to decide *which* files to measure and to fail loudly
# when it cannot. Every case below therefore exercises base resolution, not the
# length arithmetic -- that is covered by test_check_path_length.sh.
#
# Builds a throwaway upstream repository and a clone of it, so the test does not
# depend on the real tree or on network access.
#
# Usage: ./test_check_changed_path_length.sh

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CHECKER="$SCRIPT_DIR/check_changed_path_length.sh"
failures=0
tests=0

if [[ ! -x "$CHECKER" ]]; then
    echo "ERROR: $CHECKER not found or not executable"
    exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

git_q() { git -c user.email=ck@test -c user.name=ck "$@"; }

# An upstream with a develop branch, and a clone to work in. The clone carries
# the scripts under test at script/, mirroring the real layout, so the wrapper
# finds check_path_length.sh next to itself.
UP="$WORKDIR/upstream"
mkdir -p "$UP"
git -C "$UP" init --quiet --initial-branch=develop
mkdir -p "$UP/script"
cp "$SCRIPT_DIR/check_path_length.sh" "$SCRIPT_DIR/check_changed_path_length.sh" "$UP/script/"
chmod +x "$UP/script/"*.sh
echo base > "$UP/base.txt"

# A long path that is already on develop before any branch exists, so the tests
# below can distinguish "pre-existing" from "newly added".
PREEXISTING_DIR="p$(printf 'z%.0s' $(seq 1 230))"
mkdir -p "$UP/$PREEXISTING_DIR"
echo original > "$UP/$PREEXISTING_DIR/f.cpp"
PREEXISTING="$PREEXISTING_DIR/f.cpp"

git -C "$UP" add -A
git_q -C "$UP" commit --quiet -m base

REPO="$WORKDIR/repo"
git clone --quiet "$UP" "$REPO"

# Create a file whose repository-relative path is exactly $1 characters, commit
# it on the current branch, and echo the path.
add_path_of_length() {
    local target=$1 message=$2
    local dir="d" file="f.cpp"
    local pad=$(( target - ${#file} - 1 - ${#dir} ))
    (( pad >= 0 )) || { echo "target $target too small" >&2; return 1; }
    dir="${dir}$(printf 'x%.0s' $(seq 1 "$pad"))"
    mkdir -p "$REPO/$dir"
    : > "$REPO/$dir/$file"
    git -C "$REPO" add -A
    git_q -C "$REPO" commit --quiet -m "$message"
    echo "$dir/$file"
}

# check <description> <expected-exit> <args...>
check() {
    local desc=$1 expected=$2
    shift 2
    tests=$(( tests + 1 ))
    local out actual
    out=$( cd "$REPO" && "$REPO/script/check_changed_path_length.sh" "$@" 2>&1 )
    actual=$?
    if [[ "$actual" -ne "$expected" ]]; then
        echo "FAIL: $desc (expected exit $expected, got $actual)"
        [[ -n "$out" ]] && echo "$out" | sed 's/^/      /'
        failures=$(( failures + 1 ))
    else
        echo "PASS: $desc"
    fi
}

# --- a branch with nothing over the limit passes ----------------------------
git -C "$REPO" checkout --quiet -b feature
add_path_of_length 120 "short path" >/dev/null
check "clean branch passes" 0 develop

# --- a branch that adds an over-limit path fails ----------------------------
over=$(add_path_of_length 246 "long path")
check "over-limit path on the branch is rejected" 1 develop

# --- the base ref is missing and cannot be fetched: fail, do not pass --------
# This is the regression under test: the previous inline
# `git diff ... | xargs` returned xargs's status, so an unresolvable base
# produced no files and reported success while the 246-character path above
# was still present.
check "unresolvable base fails instead of passing silently" 2 no-such-branch

# --- the base ref is missing locally but fetchable: recover, then measure ----
git -C "$REPO" update-ref -d refs/remotes/origin/develop
check "missing-but-fetchable base is fetched, then the path is caught" 1 develop

# --- a long path added earlier in the branch is still seen ------------------
# The previous HEAD~1 fallback compared only the final commit, so a long path
# in an earlier commit was invisible.
git_q -C "$REPO" commit --quiet --allow-empty -m "later unrelated commit"
check "long path in an earlier commit is still caught" 1 develop

# --- deleting a long path is not blocked ------------------------------------
git -C "$REPO" rm --quiet -r "$(dirname "$over")"
git_q -C "$REPO" commit --quiet -m "remove the long path"
check "deleting a long path passes (--diff-filter=AR excludes deletions)" 0 develop

# --- MODIFYING a pre-existing long file is not blocked ----------------------
# The point of --diff-filter=AR. A change that merely edits a file that was
# already too long must not be forced into a rename project. Guard against a
# regression back to AMR.
git -C "$REPO" checkout --quiet -b modify-only origin/develop
echo "edited" >> "$REPO/$PREEXISTING"
git -C "$REPO" add -A
git_q -C "$REPO" commit --quiet -m "edit a pre-existing long file"
check "modifying a pre-existing long file passes" 0 develop

# --- but RENAMING it to another long path is still blocked ------------------
# Renames are in scope: the author is already changing the name, so this is the
# moment to make it short.
git -C "$REPO" checkout --quiet -b rename-long origin/develop
git -C "$REPO" mv "$PREEXISTING" "${PREEXISTING%.cpp}_renamed.cpp"
git_q -C "$REPO" commit --quiet -m "rename a long path to another long path"
check "renaming a long path to another long path is rejected" 1 develop

# --- renaming a long path to a SHORT one is the fix, and must pass ----------
git -C "$REPO" checkout --quiet -b rename-short origin/develop
git -C "$REPO" mv "$PREEXISTING" short_after_rename.cpp
git_q -C "$REPO" commit --quiet -m "shorten a long path"
check "renaming a long path to a short one passes" 0 develop

git -C "$REPO" checkout --quiet feature

# --- the base branch can also come from the environment ---------------------
tests=$(( tests + 1 ))
if ( cd "$REPO" && CK_BASE_BRANCH=develop "$REPO/script/check_changed_path_length.sh" >/dev/null 2>&1 ); then
    echo "PASS: CK_BASE_BRANCH supplies the base when no argument is given"
else
    echo "FAIL: CK_BASE_BRANCH supplies the base when no argument is given"
    failures=$(( failures + 1 ))
fi

echo
if [[ "$failures" -eq 0 ]]; then
    echo "All $tests checks passed."
    exit 0
fi
echo "$failures of $tests checks failed."
exit 1

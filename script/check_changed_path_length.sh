#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Runs check_path_length.sh over the files a change adds or renames.
#
# Scoped to added and renamed files rather than the whole tree because the tree
# already carries paths over the limit, and new paths are what we need to keep
# short (--diff-filter=AR).
#
# Modifications are deliberately excluded. A pull request that merely edits a
# pre-existing long file must not be forced into a rename project it did not
# sign up for: over the last year that would have ambushed a copyright-header
# sweep touching 385 long files, among others. Additions and renames catch every
# case where the backlog actually grows -- including the file that caused
# ROCM-29381, which was added, not modified.
#
# Deletions are excluded too, so removing a long path always passes.
#
# This wrapper exists so that failure to determine *which* files changed is a
# hard error rather than a silent pass. Inlined in the Jenkins stage as
#
#   git diff ... | xargs -0 -r script/check_path_length.sh
#
# a failing `git diff` produced no output, and the pipeline returned xargs's
# successful status -- the gate reported green without measuring anything. Here
# the diff is written to a file and `set -euo pipefail` is available, so the
# producer's exit status is load-bearing.
#
# Usage: ./check_changed_path_length.sh [base-branch]
#
#   base-branch   branch to compare against; defaults to $CHANGE_TARGET (set by
#                 Jenkins on PR builds), then $CK_BASE_BRANCH, then "develop".
#
# Exit codes:
#   0   no changed path exceeds the limit
#   1   at least one changed path exceeds the limit
#   2   the comparison base could not be determined (infrastructure problem,
#       not a path problem) -- deliberately not 0, so the gate cannot pass
#       without having looked at anything.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

base_branch=${1:-${CHANGE_TARGET:-${CK_BASE_BRANCH:-develop}}}
base_rev="origin/${base_branch}"

# Normally present from `checkout scm`; fetch once if it is not. A fetch failure
# is not fatal on its own -- the rev-parse below decides.
if ! git rev-parse --verify --quiet "${base_rev}^{commit}" >/dev/null; then
    echo "INFO: ${base_rev} is not present locally; fetching it." >&2
    git fetch --no-tags --quiet origin \
        "+refs/heads/${base_branch}:refs/remotes/${base_rev}" || true
fi

if ! base_sha=$(git rev-parse --verify --quiet "${base_rev}^{commit}"); then
    echo "ERROR: cannot resolve comparison base '${base_rev}'."
    echo "  The path-length check needs the target branch to decide which files"
    echo "  this change touches, and will not pass silently without it."
    echo "  Fix: fetch '${base_branch}', or pass a different base as \$1 or via"
    echo "  CK_BASE_BRANCH."
    exit 2
fi

# merge-base for both PR and branch builds: equivalent to the three-dot
# "origin/<base>...HEAD" form, and it does not silently narrow to the last
# commit the way a HEAD~1 fallback would.
if ! merge_base=$(git merge-base "$base_sha" HEAD); then
    echo "ERROR: no merge base between HEAD and ${base_rev} (${base_sha})."
    echo "  Fix: fetch the full history of '${base_branch}'; a shallow clone"
    echo "  cannot be compared against it."
    exit 2
fi

echo "INFO: comparing against ${base_rev} (merge base ${merge_base})" >&2

# A file rather than a pipe, so a failing git diff aborts under `set -e`
# instead of being masked by xargs's exit status.
changed=$(mktemp)
trap 'rm -f "$changed"' EXIT

# Pathspec '.' limits the diff to the directory this is run from
# (projects/composablekernel in the Jenkins stage); git still emits
# repository-relative paths, which is what the checker measures.
git diff -z --name-only --diff-filter=AR "$merge_base" HEAD -- . > "$changed"

# xargs reports a failing child as 123; normalise it so the three exit codes
# documented above stay distinct.
status=0
xargs -0 -r "${here}/check_path_length.sh" < "$changed" || status=$?
(( status == 0 )) || exit 1

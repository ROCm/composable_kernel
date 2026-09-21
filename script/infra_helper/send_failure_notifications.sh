#!/usr/bin/env bash
# send_failure_notifications.sh
#
# Scans the Jenkins build log for known infrastructure failure patterns and
# sends a Teams webhook notification for each match.
#
# Required environment variables (Jenkins provides all except WEBHOOK_URL):
#   BUILD_URL       - Jenkins build URL (e.g. http://host/job/foo/42/)
#   JOB_NAME        - Jenkins job name
#   BUILD_NUMBER    - Jenkins build number
#   RUN_DISPLAY_URL - Jenkins Blue Ocean display URL
#   WEBHOOK_URL     - Teams incoming webhook URL (passed via withCredentials)

# Do not echo commands — the grep command contains all pattern strings and
# would self-match if it appeared in the console log.
set +x

# ---------------------------------------------------------------------------
# Failure patterns and their descriptions (parallel indexed arrays).
# ---------------------------------------------------------------------------
PATTERNS=(
    'login attempt to .* failed with status: 401 Unauthorized'
    'docker login failed'
    'HTTP request sent .* 404 Not Found'
    '/sys/module/amdgpu/version: No such file or directory'
    'GPU not found'
    'Could not connect to Redis at .* Connection timed out'
    'unauthorized: your account must log in with a Personal Access Token'
    'sccache: error: Server startup failed: Address in use'
    'No space left on device'
    'Could not resolve host: github.com'
)

DESCRIPTIONS=(
    "Docker registry authentication failed"
    "Docker login failed"
    "HTTP request failed with 404"
    "Missing drivers"
    "GPU not found"
    "Redis connection timed out"
    "Docker login failed"
    "Sccache Error"
    "Device space error"
    "Unable to access Github"
)

# Indices into PATTERNS/DESCRIPTIONS for which a node name lookup is performed.
NODE_PATTERN_INDICES=(3 4 8 9) 

# ---------------------------------------------------------------------------
# Fetch and scan the log.
# ---------------------------------------------------------------------------
COMBINED_PATTERN=$(printf '%s\n' "${PATTERNS[@]}" | paste -sd '|')

echo "Checking for failure patterns..."
GREP_OUTPUT=$(wget -q --no-check-certificate -O - "${BUILD_URL}consoleText" \
    | grep -E -B 2 -A 2 "${COMBINED_PATTERN}" || true)

if [[ -z "$GREP_OUTPUT" ]]; then
    echo "No failure patterns found in build log"
    exit 0
fi

# ---------------------------------------------------------------------------
# Process each grep context block.
# ---------------------------------------------------------------------------
# Track descriptions already notified to avoid duplicate notifications.
declare -a NOTIFIED_DESCRIPTIONS=()

process_block() {
    local block="$1"
    [[ -z "$block" ]] && return

    for i in "${!PATTERNS[@]}"; do
        local pattern="${PATTERNS[$i]}"
        local description="${DESCRIPTIONS[$i]}"

        # Skip if this description was already notified.
        local already_notified=false
        for notified in "${NOTIFIED_DESCRIPTIONS[@]:-}"; do
            [[ "$notified" == "$description" ]] && already_notified=true && break
        done
        $already_notified && continue

        # Check if this block contains the pattern.
        if echo "$block" | grep -qE "$pattern"; then
            NOTIFIED_DESCRIPTIONS+=("$description")

            # For node-related patterns, find the most recent NODE_NAME before
            # the failure via a single forward awk pass that exits immediately
            # on the failure line, regardless of how many lines separate the two.
            local node_name=""
            for node_idx in "${NODE_PATTERN_INDICES[@]}"; do
                if [[ "$node_idx" == "$i" ]]; then
                    node_name=$(wget -q --no-check-certificate -O - "${BUILD_URL}consoleText" | awk '
                        /NODE_NAME[[:space:]]*=/ { node = $NF }
                        index($0, "'"$pattern"'") { print node; exit }
                    ')
                    break
                fi
            done

            # Escape context for safe embedding in a JSON string value:
            # backslashes first, then quotes, then newlines.
            local escaped_context
            escaped_context=$(printf '%s' "$block" \
                | sed 's/\\/\\\\/g' \
                | sed 's/"/\\"/g' \
                | sed ':a;N;$!ba;s/\n/\\n/g')

            # Build JSON payload and send notification.
            echo "Sending notification for: $description"
            {
                printf '{\n'
                printf '    "jobName": "%s",\n'      "$JOB_NAME"
                printf '    "buildNumber": "%s",\n'  "$BUILD_NUMBER"
                printf '    "jobUrl": "%s",\n'       "$RUN_DISPLAY_URL"
                printf '    "detectedIssue": "%s",\n' "$description"
                printf '    "logContext": "%s",\n'   "$escaped_context"
                printf '    "nodeName": "%s"\n'      "$node_name"
                printf '}\n'
            } > webhook_payload.json

            curl -X POST "$WEBHOOK_URL" \
                -H "Content-Type: application/json" \
                -d @webhook_payload.json

            rm -f webhook_payload.json
        fi
    done
}

# grep separates non-adjacent match groups with a line containing just "--".
# Read line by line, accumulate into a block, and process when the separator
# is hit. The final block has no trailing "--" so it is processed after the loop.
current_block=""
while IFS= read -r line; do
    if [[ "$line" == "--" ]]; then
        process_block "$current_block"
        current_block=""
    else
        current_block+="$line"$'\n'
    fi
done <<< "$GREP_OUTPUT"
process_block "$current_block"

echo "Done failure pattern checking and notifications"

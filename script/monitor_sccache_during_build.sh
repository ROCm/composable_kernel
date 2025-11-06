#!/bin/bash
# Continuous monitoring script for sccache during builds
# Usage: ./monitor_sccache_during_build.sh [log_prefix] &

LOG_PREFIX=${1:-"sccache_monitor"}
MONITOR_LOG="${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
MONITOR_INTERVAL=30  # seconds

echo "Starting sccache monitoring - logging to $MONITOR_LOG"
echo "Monitor interval: $MONITOR_INTERVAL seconds"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

# Function to get sccache stats safely
get_sccache_stats() {
    if command -v sccache &> /dev/null; then
        sccache --show-stats 2>/dev/null || echo "sccache stats unavailable"
    else
        echo "sccache command not found"
    fi
}

# Function to test Redis connectivity
test_redis_connectivity() {
    if [ -n "${SCCACHE_REDIS}" ]; then
        local start_time=$(date +%s%N)
        local response=$(timeout 5 redis-cli -u "${SCCACHE_REDIS}" ping 2>&1) || response="TIMEOUT"
        local end_time=$(date +%s%N)
        local latency=$(( (end_time - start_time) / 1000000 ))
        echo "Redis: $response (${latency}ms)"
    else
        echo "Redis: SCCACHE_REDIS not set"
    fi
}

log_with_timestamp "=== SCCACHE MONITORING STARTED ==="
log_with_timestamp "PID: $$"
log_with_timestamp "Node: ${NODE_NAME:-$(hostname)}"
log_with_timestamp "SCCACHE_REDIS: ${SCCACHE_REDIS:-not set}"

# Initial state
log_with_timestamp "=== INITIAL STATE ==="
log_with_timestamp "$(get_sccache_stats)"
log_with_timestamp "$(test_redis_connectivity)"

# Monitor loop
while true; do
    sleep $MONITOR_INTERVAL
    
    # Check if sccache server is still running
    if ! pgrep -f "sccache.*--start-server" > /dev/null; then
        log_with_timestamp "WARNING: sccache server not running!"
    fi
    
    # Get current stats
    current_stats=$(get_sccache_stats)
    redis_status=$(test_redis_connectivity)
    
    # Extract cache hit information
    cache_hits=$(echo "$current_stats" | grep -E "(Cache hits|Compile requests)" | tr '\n' ' ')
    
    log_with_timestamp "Stats: $cache_hits | $redis_status"
    
    # Check for Redis latency issues
    if echo "$redis_status" | grep -E "[0-9]{3,}" > /dev/null; then  # >100ms latency
        log_with_timestamp "HIGH REDIS LATENCY detected"
    fi
    
    # Check for Redis connection failures
    if echo "$redis_status" | grep -E "(TIMEOUT|Connection refused|No route)" > /dev/null; then
        log_with_timestamp "REDIS CONNECTION FAILURE detected"
    fi
done
#!/bin/bash
# Debug script to analyze sccache performance issues
# Usage: ./debug_sccache_performance.sh [build_id]

BUILD_ID=${1:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE="sccache_debug_${BUILD_ID}.log"

echo "=== SCCACHE PERFORMANCE DEBUG - ${BUILD_ID} ===" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Node: ${NODE_NAME:-$(hostname)}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if sccache is running
echo "=== SCCACHE SERVER STATUS ===" | tee -a "$LOG_FILE"
if pgrep -f sccache > /dev/null; then
    echo "sccache server is running" | tee -a "$LOG_FILE"
    ps aux | grep sccache | grep -v grep | tee -a "$LOG_FILE"
else
    echo "sccache server is NOT running" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Get comprehensive sccache statistics
echo "=== SCCACHE STATISTICS ===" | tee -a "$LOG_FILE"
if command -v sccache &> /dev/null; then
    sccache --show-stats 2>&1 | tee -a "$LOG_FILE"
else
    echo "sccache command not found" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Check Redis connectivity and performance
echo "=== REDIS CONNECTIVITY TEST ===" | tee -a "$LOG_FILE"
if [ -n "${SCCACHE_REDIS}" ]; then
    echo "Redis URL: ${SCCACHE_REDIS}" | tee -a "$LOG_FILE"
    
    # Test basic connectivity
    start_time=$(date +%s%N)
    redis_response=$(redis-cli -u "${SCCACHE_REDIS}" ping 2>&1) || redis_response="FAILED"
    end_time=$(date +%s%N)
    latency=$(( (end_time - start_time) / 1000000 ))
    
    echo "Redis ping response: ${redis_response}" | tee -a "$LOG_FILE"
    echo "Redis ping latency: ${latency}ms" | tee -a "$LOG_FILE"
    
    # Test Redis performance with larger operation
    echo "Testing Redis write/read performance..." | tee -a "$LOG_FILE"
    start_time=$(date +%s%N)
    redis-cli -u "${SCCACHE_REDIS}" set "test_key_${BUILD_ID}" "test_value_$(date)" >/dev/null 2>&1
    redis-cli -u "${SCCACHE_REDIS}" get "test_key_${BUILD_ID}" >/dev/null 2>&1
    redis-cli -u "${SCCACHE_REDIS}" del "test_key_${BUILD_ID}" >/dev/null 2>&1
    end_time=$(date +%s%N)
    redis_perf_latency=$(( (end_time - start_time) / 1000000 ))
    echo "Redis write/read/delete latency: ${redis_perf_latency}ms" | tee -a "$LOG_FILE"
    
else
    echo "SCCACHE_REDIS environment variable not set" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Check Redis memory status
echo "=== REDIS MEMORY STATUS ===" | tee -a "$LOG_FILE"
if [ -n "${SCCACHE_REDIS}" ]; then
    redis-cli -u "${SCCACHE_REDIS}" info memory 2>&1 | grep -E "(used_memory|maxmemory|evicted_keys|keyspace)" | tee -a "$LOG_FILE"
else
    echo "Cannot check Redis memory - SCCACHE_REDIS not set" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Check compiler fingerprint consistency
echo "=== COMPILER FINGERPRINT ===" | tee -a "$LOG_FILE"
if [ -f "${SCCACHE_EXTRAFILES}" ]; then
    echo "Compiler fingerprint file: ${SCCACHE_EXTRAFILES}" | tee -a "$LOG_FILE"
    cat "${SCCACHE_EXTRAFILES}" | tee -a "$LOG_FILE"
else
    echo "Compiler fingerprint file not found: ${SCCACHE_EXTRAFILES}" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Environment variables affecting sccache
echo "=== SCCACHE ENVIRONMENT ===" | tee -a "$LOG_FILE"
env | grep -E "(SCCACHE|ROCM|HIP)" | sort | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== DEBUG COMPLETE ===" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
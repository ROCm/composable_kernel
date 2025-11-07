#!/bin/bash
# Debug script to analyze sccache performance issues
# Usage: ./debug_sccache_performance.sh [build_id]

BUILD_ID=${1:-$(date +%Y%m%d_%H%M%S)}

# Include stage name in log filename if available
STAGE_SUFFIX=""
if [ -n "${JENKINS_STAGE_NAME}" ]; then
    # Convert stage name to filename-safe format (replace spaces and special chars with underscores)
    STAGE_SAFE=$(echo "${JENKINS_STAGE_NAME}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g')
    STAGE_SUFFIX="_${STAGE_SAFE}"
fi

LOG_FILE="logs/sccache_debug_${BUILD_ID}${STAGE_SUFFIX}.log"

echo "=== SCCACHE PERFORMANCE DEBUG - ${BUILD_ID} ===" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Node: ${NODE_NAME:-$(hostname)}" | tee -a "$LOG_FILE"
echo "Stage: ${JENKINS_STAGE_NAME:-unknown}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if sccache is running
echo "=== SCCACHE SERVER STATUS ===" | tee -a "$LOG_FILE"
if command -v sccache &> /dev/null; then
    if sccache --show-stats &> /dev/null; then
        echo "sccache server is running" | tee -a "$LOG_FILE"
        ps aux | grep sccache | grep -v grep | tee -a "$LOG_FILE"
    else
        echo "sccache server is NOT running" | tee -a "$LOG_FILE"
        echo "Attempting to start sccache server..." | tee -a "$LOG_FILE"
        sccache --start-server 2>&1 | tee -a "$LOG_FILE"
    fi
else
    echo "sccache command not found" | tee -a "$LOG_FILE"
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

# Use SCCACHE_REDIS if set, otherwise construct from CK_SCCACHE
if [ -n "${SCCACHE_REDIS}" ]; then
    REDIS_URL="${SCCACHE_REDIS}"
elif [ -n "${CK_SCCACHE}" ]; then
    REDIS_URL="redis://${CK_SCCACHE}"
else
    REDIS_URL=""
fi

if [ -n "${REDIS_URL}" ]; then
    echo "Redis URL: ${REDIS_URL}" | tee -a "$LOG_FILE"
    
    # Test basic connectivity
    start_time=$(date +%s%N)
    redis_response=$(redis-cli -u "${REDIS_URL}" ping 2>&1) || redis_response="FAILED"
    end_time=$(date +%s%N)
    latency=$(( (end_time - start_time) / 1000000 ))
    
    echo "Redis ping response: ${redis_response}" | tee -a "$LOG_FILE"
    echo "Redis ping latency: ${latency}ms" | tee -a "$LOG_FILE"
    
    # Test Redis performance with larger operation
    echo "Testing Redis write/read performance..." | tee -a "$LOG_FILE"
    start_time=$(date +%s%N)
    redis-cli -u "${REDIS_URL}" set "test_key_${BUILD_ID}" "test_value_$(date)" >/dev/null 2>&1
    redis-cli -u "${REDIS_URL}" get "test_key_${BUILD_ID}" >/dev/null 2>&1
    redis-cli -u "${REDIS_URL}" del "test_key_${BUILD_ID}" >/dev/null 2>&1
    end_time=$(date +%s%N)
    redis_perf_latency=$(( (end_time - start_time) / 1000000 ))
    echo "Redis write/read/delete latency: ${redis_perf_latency}ms" | tee -a "$LOG_FILE"
    
else
    echo "No Redis URL available (neither SCCACHE_REDIS nor CK_SCCACHE set)" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Check Redis memory status
echo "=== REDIS MEMORY STATUS ===" | tee -a "$LOG_FILE"
if [ -n "${REDIS_URL}" ]; then
    redis-cli -u "${REDIS_URL}" info memory 2>&1 | grep -E "(used_memory|maxmemory|evicted_keys|keyspace)" | tee -a "$LOG_FILE"
else
    echo "Cannot check Redis memory - no Redis URL available" | tee -a "$LOG_FILE"
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

# Cache key components analysis
echo "=== CACHE KEY COMPONENTS ANALYSIS ===" | tee -a "$LOG_FILE"

# 1. Compiler binary hash
if [ -f "${ROCM_PATH}/bin/hipcc" ]; then
    HIPCC_HASH=$(md5sum "${ROCM_PATH}/bin/hipcc" | cut -d' ' -f1)
    echo "hipcc binary hash: ${HIPCC_HASH}" | tee -a "$LOG_FILE"
else
    echo "hipcc binary not found at ${ROCM_PATH}/bin/hipcc" | tee -a "$LOG_FILE"
fi

# 2. Custom cache buster (architecture-specific)
echo "SCCACHE_C_CUSTOM_CACHE_BUSTER: ${SCCACHE_C_CUSTOM_CACHE_BUSTER:-not set}" | tee -a "$LOG_FILE"

# 3. Extra files (compiler fingerprint)
if [ -f "${SCCACHE_EXTRAFILES}" ]; then
    EXTRAFILES_HASH=$(md5sum "${SCCACHE_EXTRAFILES}" | cut -d' ' -f1)
    echo "SCCACHE_EXTRAFILES hash: ${EXTRAFILES_HASH}" | tee -a "$LOG_FILE"
    echo "SCCACHE_EXTRAFILES content hash components:" | tee -a "$LOG_FILE"
    while IFS= read -r line; do
        echo "  $line" | tee -a "$LOG_FILE"
    done < "${SCCACHE_EXTRAFILES}"
else
    echo "SCCACHE_EXTRAFILES not found" | tee -a "$LOG_FILE"
fi

# 4. Current working directory and build flags
echo "Current working directory: $(pwd)" | tee -a "$LOG_FILE"
echo "CMAKE flags that affect cache keys:" | tee -a "$LOG_FILE"
env | grep -E "(CMAKE_|CXXFLAGS|CFLAGS)" | sort | tee -a "$LOG_FILE"

# 5. Test cache key generation with a simple file
echo "=== CACHE KEY TEST ===" | tee -a "$LOG_FILE"
cat > /tmp/test_cache_key.cpp << 'EOF'
#include <iostream>
int main() { return 0; }
EOF

if command -v sccache &> /dev/null && [ -n "${SCCACHE_REDIS}" ]; then
    echo "Testing cache key generation with simple file..." | tee -a "$LOG_FILE"
    # Enable verbose logging temporarily
    export SCCACHE_LOG=trace
    timeout 30 sccache hipcc -c /tmp/test_cache_key.cpp -o /tmp/test_cache_key.o 2>&1 | grep -E "(cache key|Cache key)" | head -5 | tee -a "$LOG_FILE"
    unset SCCACHE_LOG
    rm -f /tmp/test_cache_key.cpp /tmp/test_cache_key.o
else
    echo "Cannot test cache key - sccache not available or Redis not configured" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "=== DEBUG COMPLETE ===" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
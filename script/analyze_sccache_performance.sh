#!/bin/bash
# Analysis script to compare fast vs slow sccache builds
# Usage: ./analyze_sccache_performance.sh [fast_log] [slow_log]

FAST_LOG=$1
SLOW_LOG=$2

if [ -z "$FAST_LOG" ] || [ -z "$SLOW_LOG" ]; then
    echo "Usage: $0 <fast_build_log> <slow_build_log>"
    echo ""
    echo "This script compares sccache debug logs from fast and slow builds"
    echo "to identify what's causing performance differences."
    exit 1
fi

if [ ! -f "$FAST_LOG" ] || [ ! -f "$SLOW_LOG" ]; then
    echo "Error: One or both log files not found"
    echo "Fast log: $FAST_LOG"
    echo "Slow log: $SLOW_LOG"
    exit 1
fi

echo "=== SCCACHE PERFORMANCE COMPARISON ==="
echo "Fast build log: $FAST_LOG"
echo "Slow build log: $SLOW_LOG"

# Extract stage information from filenames if available
FAST_STAGE=$(echo "$FAST_LOG" | sed -n 's/.*_\([^_]*\)\.log$/\1/p')
SLOW_STAGE=$(echo "$SLOW_LOG" | sed -n 's/.*_\([^_]*\)\.log$/\1/p')

if [ -n "$FAST_STAGE" ] && [ -n "$SLOW_STAGE" ]; then
    echo "Fast build stage: $FAST_STAGE"
    echo "Slow build stage: $SLOW_STAGE"
    if [ "$FAST_STAGE" != "$SLOW_STAGE" ]; then
        echo "WARNING: Comparing logs from different stages!"
    fi
fi
echo ""

# Function to extract value from log
extract_value() {
    local log_file=$1
    local pattern=$2
    grep "$pattern" "$log_file" | head -1 | sed 's/.*: //'
}

# Function to extract sccache stats
extract_stats() {
    local log_file=$1
    echo "=== Stats from $log_file ==="
    grep -A 20 "SCCACHE STATISTICS" "$log_file" | head -25
    echo ""
}

echo "=== CACHE HIT COMPARISON ==="
fast_hits=$(extract_value "$FAST_LOG" "Cache hits")
slow_hits=$(extract_value "$SLOW_LOG" "Cache hits")
echo "Fast build cache hits: $fast_hits"
echo "Slow build cache hits: $slow_hits"
echo ""

echo "=== CACHE MISS COMPARISON ==="
fast_misses=$(extract_value "$FAST_LOG" "Cache misses")
slow_misses=$(extract_value "$SLOW_LOG" "Cache misses")
echo "Fast build cache misses: $fast_misses"
echo "Slow build cache misses: $slow_misses"
echo ""

echo "=== REDIS LATENCY COMPARISON ==="
fast_latency=$(extract_value "$FAST_LOG" "Redis ping latency")
slow_latency=$(extract_value "$SLOW_LOG" "Redis ping latency")
echo "Fast build Redis latency: $fast_latency"
echo "Slow build Redis latency: $slow_latency"
echo ""

echo "=== COMPILER FINGERPRINT COMPARISON ==="
echo "--- Fast build fingerprint ---"
sed -n '/COMPILER FINGERPRINT/,/===/p' "$FAST_LOG" | head -10
echo ""
echo "--- Slow build fingerprint ---"
sed -n '/COMPILER FINGERPRINT/,/===/p' "$SLOW_LOG" | head -10
echo ""

echo "=== REDIS MEMORY COMPARISON ==="
echo "--- Fast build Redis memory ---"
grep -A 5 "REDIS MEMORY STATUS" "$FAST_LOG"
echo ""
echo "--- Slow build Redis memory ---"
grep -A 5 "REDIS MEMORY STATUS" "$SLOW_LOG"
echo ""

echo "=== ENVIRONMENT COMPARISON ==="
echo "--- Fast build environment ---"
grep -A 10 "SCCACHE ENVIRONMENT" "$FAST_LOG" | head -15
echo ""
echo "--- Slow build environment ---"
grep -A 10 "SCCACHE ENVIRONMENT" "$SLOW_LOG" | head -15
echo ""

# Check for specific issues
echo "=== ISSUE DETECTION ==="

# Check for Redis connectivity issues
if grep -q "FAILED\|TIMEOUT\|Connection refused" "$SLOW_LOG"; then
    echo "REDIS CONNECTIVITY ISSUES detected in slow build"
else
    echo "No Redis connectivity issues in slow build"
fi

# Check for sccache server issues
if grep -q "sccache server is NOT running" "$SLOW_LOG"; then
    echo "SCCACHE SERVER DOWN detected in slow build"
else
    echo "sccache server running in slow build"
fi

# Check for high latency
slow_latency_num=$(echo "$slow_latency" | grep -o '[0-9]*' | head -1)
if [ -n "$slow_latency_num" ] && [ "$slow_latency_num" -gt 100 ]; then
    echo "HIGH REDIS LATENCY detected in slow build (${slow_latency_num}ms)"
else
    echo "Redis latency acceptable in slow build"
fi

# Check for cache key differences
echo ""
echo "=== CACHE KEY COMPONENT ANALYSIS ==="

# Compare compiler hashes
fast_hipcc_hash=$(grep "hipcc binary hash:" "$FAST_LOG" | cut -d' ' -f4)
slow_hipcc_hash=$(grep "hipcc binary hash:" "$SLOW_LOG" | cut -d' ' -f4)
if [ -n "$fast_hipcc_hash" ] && [ -n "$slow_hipcc_hash" ]; then
    if [ "$fast_hipcc_hash" != "$slow_hipcc_hash" ]; then
        echo "DIFFERENT HIPCC BINARY HASHES - This will cause cache misses!"
        echo "   Fast: $fast_hipcc_hash"
        echo "   Slow: $slow_hipcc_hash"
    else
        echo "hipcc binary hashes are identical"
    fi
else
    echo "Could not find hipcc binary hashes in logs"
fi

# Compare cache busters
fast_buster=$(grep "SCCACHE_C_CUSTOM_CACHE_BUSTER:" "$FAST_LOG" | cut -d' ' -f2-)
slow_buster=$(grep "SCCACHE_C_CUSTOM_CACHE_BUSTER:" "$SLOW_LOG" | cut -d' ' -f2-)
if [ -n "$fast_buster" ] && [ -n "$slow_buster" ]; then
    if [ "$fast_buster" != "$slow_buster" ]; then
        echo "DIFFERENT CACHE BUSTERS - This will cause cache misses!"
        echo "   Fast: $fast_buster"
        echo "   Slow: $slow_buster"
    else
        echo "Cache busters are identical"
    fi
else
    echo "Could not find cache busters in logs"
fi

# Compare extra files hashes
fast_extra_hash=$(grep "SCCACHE_EXTRAFILES hash:" "$FAST_LOG" | cut -d' ' -f3)
slow_extra_hash=$(grep "SCCACHE_EXTRAFILES hash:" "$SLOW_LOG" | cut -d' ' -f3)
if [ -n "$fast_extra_hash" ] && [ -n "$slow_extra_hash" ]; then
    if [ "$fast_extra_hash" != "$slow_extra_hash" ]; then
        echo "DIFFERENT EXTRAFILES HASHES - This will cause cache misses!"
        echo "   Fast: $fast_extra_hash"  
        echo "   Slow: $slow_extra_hash"
        echo "   This means compiler fingerprints are different between builds"
    else
        echo "SCCACHE_EXTRAFILES hashes are identical"
    fi
else
    echo "Could not find SCCACHE_EXTRAFILES hashes in logs"
fi

echo ""
echo "=== CACHE KEY DIAGNOSIS ==="
echo "If all cache key components are identical but you still have cache misses,"
echo "the problem is likely:"
echo "1. Redis connectivity issues (timeouts, network problems)"
echo "2. sccache server crashes or restarts"
echo "3. Source file changes (even whitespace/comments affect cache keys)"
echo "4. Include file changes (headers that are #included)"
echo ""
echo "If cache key components are different, that explains the cache misses."
echo "Look at the specific differences above to understand why."

echo ""
echo "=== DETAILED STATISTICS COMPARISON ==="
extract_stats "$FAST_LOG"
extract_stats "$SLOW_LOG"
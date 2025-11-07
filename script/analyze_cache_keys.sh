#!/bin/bash
# sccache Cache Key Analysis Script
# This script helps determine why sccache might be generating different cache keys
# for what should be identical compilation units

BUILD_ID=${1:-$(date +%Y%m%d_%H%M%S)}

# Include stage name in log filename if available
STAGE_SUFFIX=""
if [ -n "${JENKINS_STAGE_NAME}" ]; then
    # Convert stage name to filename-safe format (replace spaces and special chars with underscores)
    STAGE_SAFE=$(echo "${JENKINS_STAGE_NAME}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g')
    STAGE_SUFFIX="_${STAGE_SAFE}"
fi

CACHE_KEY_LOG="logs/cache_key_analysis_${BUILD_ID}${STAGE_SUFFIX}.log"

echo "=== SCCACHE CACHE KEY ANALYSIS ===" | tee "$CACHE_KEY_LOG"
echo "Timestamp: $(date)" | tee -a "$CACHE_KEY_LOG"
echo "Build ID: $BUILD_ID" | tee -a "$CACHE_KEY_LOG"
echo "Stage: ${JENKINS_STAGE_NAME:-unknown}" | tee -a "$CACHE_KEY_LOG"
echo "" | tee -a "$CACHE_KEY_LOG"

# Function to generate hash of all cache key components
generate_cache_key_fingerprint() {
    echo "=== CACHE KEY FINGERPRINT GENERATION ===" | tee -a "$CACHE_KEY_LOG"
    
    # Create a deterministic hash of all components that affect cache keys
    {
        # 1. Compiler binary
        if [ -f "${ROCM_PATH}/bin/hipcc" ]; then
            echo "HIPCC_BINARY:"
            md5sum "${ROCM_PATH}/bin/hipcc"
        fi
        
        # 2. Compiler version info
        if [ -f "${ROCM_PATH}/llvm/bin/clang" ]; then
            echo "CLANG_VERSION:"
            "${ROCM_PATH}/llvm/bin/clang" --version | head -1
        fi
        
        # 3. Extra files content
        if [ -f "${SCCACHE_EXTRAFILES}" ]; then
            echo "SCCACHE_EXTRAFILES:"
            cat "${SCCACHE_EXTRAFILES}" | sort
        fi
        
        # 4. Custom cache buster
        echo "CACHE_BUSTER:"
        echo "${SCCACHE_C_CUSTOM_CACHE_BUSTER:-none}"
        
        # 5. Environment variables that affect compilation
        echo "BUILD_ENV:"
        env | grep -E "(ROCM_PATH|HIP_PATH|CMAKE_|CXXFLAGS|CFLAGS|LDFLAGS)" | sort
        
        # 6. ROCm devicelib bitcodes
        if [ -d "${ROCM_PATH}/amdgcn/bitcode" ]; then
            echo "DEVICELIB_BITCODES:"
            find "${ROCM_PATH}/amdgcn/bitcode" -type f -name "*.bc" -exec md5sum {} \; | sort
        fi
        
    } | md5sum | tee -a "$CACHE_KEY_LOG"
}

# Function to test cache behavior with identical files
test_cache_consistency() {
    echo "=== CACHE CONSISTENCY TEST ===" | tee -a "$CACHE_KEY_LOG"
    
    # Create identical test files
    TEST_DIR="/tmp/sccache_cache_test_$$"
    mkdir -p "$TEST_DIR"
    
    cat > "$TEST_DIR/test1.cpp" << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>
int main() {
    int device_count;
    hipGetDeviceCount(&device_count);
    std::cout << "Devices: " << device_count << std::endl;
    return 0;
}
EOF
    
    # Create identical copy
    cp "$TEST_DIR/test1.cpp" "$TEST_DIR/test2.cpp"
    
    if command -v sccache &> /dev/null; then
        echo "Compiling identical files to test cache behavior..." | tee -a "$CACHE_KEY_LOG"
        
        # Clear any existing stats
        sccache --zero-stats >/dev/null 2>&1
        
        # Compile first file
        echo "Compiling test1.cpp..." | tee -a "$CACHE_KEY_LOG"
        sccache hipcc -c "$TEST_DIR/test1.cpp" -o "$TEST_DIR/test1.o" 2>&1 | tee -a "$CACHE_KEY_LOG"
        
        # Get stats after first compilation
        echo "Stats after first compilation:" | tee -a "$CACHE_KEY_LOG"
        sccache --show-stats | tee -a "$CACHE_KEY_LOG"
        
        # Compile identical second file
        echo "Compiling identical test2.cpp..." | tee -a "$CACHE_KEY_LOG"
        sccache hipcc -c "$TEST_DIR/test2.cpp" -o "$TEST_DIR/test2.o" 2>&1 | tee -a "$CACHE_KEY_LOG"
        
        # Get final stats
        echo "Stats after second compilation:" | tee -a "$CACHE_KEY_LOG"
        sccache --show-stats | tee -a "$CACHE_KEY_LOG"
        
        # Check if we got a cache hit
        CACHE_HITS=$(sccache --show-stats | grep "Cache hits" | grep -o '[0-9]*' | head -1)
        if [ "$CACHE_HITS" -gt 0 ]; then
            echo "CACHE HIT achieved for identical files" | tee -a "$CACHE_KEY_LOG"
        else
            echo "NO CACHE HIT for identical files - cache key problem!" | tee -a "$CACHE_KEY_LOG"
        fi
    else
        echo "sccache command not available" | tee -a "$CACHE_KEY_LOG"
    fi
    
    # Cleanup
    rm -rf "$TEST_DIR"
}

# Function to extract and analyze actual cache keys from sccache logs
analyze_sccache_logs() {
    echo "=== SCCACHE LOG ANALYSIS ===" | tee -a "$CACHE_KEY_LOG"
    
    # Enable detailed sccache logging
    export SCCACHE_LOG=trace
    export RUST_LOG=sccache=trace
    
    # Create a simple test file
    TEST_FILE="/tmp/cache_key_test_$$.cpp"
    cat > "$TEST_FILE" << 'EOF'
#include <iostream>
int main() { return 0; }
EOF
    
    echo "Compiling with detailed logging to capture cache key..." | tee -a "$CACHE_KEY_LOG"
    
    # Compile and capture logs
    timeout 30 sccache hipcc -c "$TEST_FILE" -o "${TEST_FILE}.o" 2>&1 | \
        grep -E "(cache key|Cache key|key.*=|hash.*=)" | \
        head -10 | tee -a "$CACHE_KEY_LOG"
    
    # Cleanup
    rm -f "$TEST_FILE" "${TEST_FILE}.o"
    unset SCCACHE_LOG RUST_LOG
}

# Function to compare two cache key fingerprints
compare_cache_keys() {
    if [ -n "$1" ] && [ -f "$1" ]; then
        echo "=== COMPARING WITH PREVIOUS BUILD ===" | tee -a "$CACHE_KEY_LOG"
        echo "Previous build log: $1" | tee -a "$CACHE_KEY_LOG"
        
        # Extract fingerprints
        CURRENT_FINGERPRINT=$(grep "CACHE KEY FINGERPRINT" -A 20 "$CACHE_KEY_LOG" | tail -1)
        PREVIOUS_FINGERPRINT=$(grep "CACHE KEY FINGERPRINT" -A 20 "$1" | tail -1)
        
        echo "Current fingerprint:  $CURRENT_FINGERPRINT" | tee -a "$CACHE_KEY_LOG"
        echo "Previous fingerprint: $PREVIOUS_FINGERPRINT" | tee -a "$CACHE_KEY_LOG"
        
        if [ "$CURRENT_FINGERPRINT" = "$PREVIOUS_FINGERPRINT" ]; then
            echo "Cache key components are IDENTICAL" | tee -a "$CACHE_KEY_LOG"
        else
            echo "Cache key components are DIFFERENT" | tee -a "$CACHE_KEY_LOG"
            echo "This explains why sccache is not reusing cached results!" | tee -a "$CACHE_KEY_LOG"
        fi
    fi
}

# Run all analysis functions
generate_cache_key_fingerprint
test_cache_consistency
analyze_sccache_logs
compare_cache_keys "$2"  # Optional: pass previous log file as second argument

echo "" | tee -a "$CACHE_KEY_LOG"
echo "=== ANALYSIS COMPLETE ===" | tee -a "$CACHE_KEY_LOG"
echo "Cache key analysis saved to: $CACHE_KEY_LOG" | tee -a "$CACHE_KEY_LOG"
echo "" | tee -a "$CACHE_KEY_LOG"
echo "INTERPRETATION GUIDE:" | tee -a "$CACHE_KEY_LOG"
echo "- If cache key fingerprints are identical between fast/slow builds," | tee -a "$CACHE_KEY_LOG"
echo "  the problem is likely Redis connectivity or sccache server issues" | tee -a "$CACHE_KEY_LOG"
echo "- If cache key fingerprints are different, look for:" | tee -a "$CACHE_KEY_LOG"
echo "  * Different compiler binary hashes" | tee -a "$CACHE_KEY_LOG"
echo "  * Different SCCACHE_C_CUSTOM_CACHE_BUSTER values" | tee -a "$CACHE_KEY_LOG"
echo "  * Different environment variables" | tee -a "$CACHE_KEY_LOG"
echo "  * Different compiler fingerprint file contents" | tee -a "$CACHE_KEY_LOG"
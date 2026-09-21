#!/bin/bash
# Local Smart Build Runner for ComposableKernel
# Run smart build workflow locally without Jenkins

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PARALLEL=$(nproc)
BASE_REF="HEAD~1"  # Previous commit (default for local testing)
TARGET_REF="HEAD"  # Current state including uncommitted changes
BUILD_DIR="../../build"
CTEST_ONLY="--ctest-only"
WORKSPACE_ROOT="../.."

print_help() {
    cat << 'HELP'
Usage: local_smart_build.sh [COMMAND] [OPTIONS]

Commands:
  analyze     Generate dependency map (step 1)
  select      Select affected tests (step 2)
  build       Build selected tests (step 3)
  test        Run selected tests (step 4)
  all         Run complete workflow (analyze → select → build → test)
  stats       Show statistics about test selection
  clean       Clean generated files

Options:
  -b, --base-ref REF          Base ref to compare against (default: HEAD~1)
  -t, --target-ref REF        Target ref to compare (default: HEAD)
  -j, --parallel NUM          Parallel jobs for analysis (default: nproc)
  --build-dir DIR             Build directory relative to script (default: ../../build)
  --no-ctest-only             Include all executables (benchmarks, examples)
  -h, --help                  Show this help

Examples:
  # Test uncommitted changes vs last commit (default)
  ./local_smart_build.sh all

  # Test current branch vs develop
  ./local_smart_build.sh select -b origin/develop -t HEAD

  # Test specific commit range
  ./local_smart_build.sh all -b abc123 -t def456

  # Step by step
  ./local_smart_build.sh analyze
  ./local_smart_build.sh select
  ./local_smart_build.sh build
  ./local_smart_build.sh test

  # Include all executables (not just tests)
  ./local_smart_build.sh all --no-ctest-only

Default behavior (no options):
  Compares HEAD~1 (previous commit) vs HEAD (current state + uncommitted changes)
  This tests your latest changes including work-in-progress.

File locations (in build directory):
  - compile_commands.json              (CMake generated)
  - build.ninja                        (CMake generated)
  - enhanced_dependency_mapping.json   (analyze output)
  - tests_to_run.json                  (select output)
HELP
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v cmake &> /dev/null; then
        log_error "cmake not found. Please install CMake."
        exit 1
    fi
    
    if ! command -v ninja &> /dev/null; then
        log_error "ninja not found. Please install Ninja build system."
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python 3."
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Please install jq for JSON processing."
        exit 1
    fi
    
    log_info "All prerequisites found ✓"
}

cmd_analyze() {
    log_info "Step 1: Generating dependency map..."

    cd "$BUILD_DIR" || exit 1

    # Always reconfigure CMake to ensure fresh compile_commands.json
    log_info "Running CMake configure to generate fresh compile_commands.json..."

    # Use CMAKE flags similar to the dev preset and README recommendations
    cmake -G Ninja \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DBUILD_DEV=ON \
        "$WORKSPACE_ROOT"

    if [ ! -f "build.ninja" ]; then
        log_error "build.ninja not found after CMake configure"
        exit 1
    fi

    log_info "Analyzing dependencies with $PARALLEL workers (this takes ~2 minutes)..."
    python3 "$WORKSPACE_ROOT/script/dependency-parser/main.py" cmake-parse \
        compile_commands.json \
        build.ninja \
        --workspace-root "$WORKSPACE_ROOT" \
        --parallel "$PARALLEL" \
        --output enhanced_dependency_mapping.json

    log_info "Dependency map generated: enhanced_dependency_mapping.json ✓"

    # Show stats
    local num_files=$(jq '.file_to_executables | length' enhanced_dependency_mapping.json)
    log_info "Mapped $num_files files to executables"
}

cmd_select() {
    log_info "Step 2: Selecting affected tests..."
    
    cd "$BUILD_DIR" || exit 1
    
    if [ ! -f "enhanced_dependency_mapping.json" ]; then
        log_error "Dependency map not found. Run 'analyze' first."
        exit 1
    fi
    
    log_info "Comparing $BASE_REF → $TARGET_REF..."
    python3 "$WORKSPACE_ROOT/script/dependency-parser/main.py" select \
        enhanced_dependency_mapping.json \
        "$BASE_REF" \
        "$TARGET_REF" \
        $CTEST_ONLY \
        --output tests_to_run.json
    
    # Show statistics
    local num_files=$(jq -r '.statistics.total_changed_files' tests_to_run.json)
    local num_tests=$(jq -r '.statistics.total_affected_executables' tests_to_run.json)
    local num_chunks=$(jq -r '.statistics.num_regex_chunks' tests_to_run.json)
    
    log_info "Test selection complete ✓"
    echo "  Changed files: $num_files"
    echo "  Affected tests: $num_tests"
    echo "  Regex chunks: $num_chunks"
    
    if [ "$num_tests" -eq 0 ]; then
        log_warn "No tests affected by your changes"
    fi
}

cmd_build() {
    log_info "Step 3: Building affected tests..."
    
    cd "$BUILD_DIR" || exit 1
    
    if [ ! -f "tests_to_run.json" ]; then
        log_error "Test selection not found. Run 'select' first."
        exit 1
    fi
    
    local num_tests=$(jq -r '.statistics.total_affected_executables' tests_to_run.json)
    
    if [ "$num_tests" -eq 0 ]; then
        log_warn "No tests to build"
        return 0
    fi
    
    log_info "Building $num_tests test executables..."
    
    local targets=$(jq -r '.executables[]' tests_to_run.json | tr '\n' ' ')
    
    if [ -n "$targets" ]; then
        ninja -j"$PARALLEL" $targets
        log_info "Build complete ✓"
    else
        log_warn "No targets to build"
    fi
}

cmd_test() {
    log_info "Step 4: Running affected tests..."
    
    cd "$BUILD_DIR" || exit 1
    
    if [ ! -f "tests_to_run.json" ]; then
        log_error "Test selection not found. Run 'select' first."
        exit 1
    fi
    
    local num_chunks=$(jq -r '.regex_chunks | length' tests_to_run.json)
    
    if [ "$num_chunks" -eq 0 ]; then
        log_warn "No tests to run"
        return 0
    fi
    
    log_info "Running tests in $num_chunks chunk(s)..."
    
    if [ "$num_chunks" -eq 1 ]; then
        # Single chunk - simple case
        local regex=$(jq -r '.regex_chunks[0]' tests_to_run.json)
        ctest -R "$regex" --output-on-failure
    else
        # Multiple chunks
        for i in $(seq 0 $((num_chunks - 1))); do
            log_info "Running test chunk $((i + 1))/$num_chunks"
            local regex=$(jq -r ".regex_chunks[$i]" tests_to_run.json)
            ctest -R "$regex" --output-on-failure
        done
    fi
    
    log_info "All tests complete ✓"
}

cmd_all() {
    log_info "Running complete smart build workflow..."
    log_info "Testing changes: $BASE_REF → $TARGET_REF"
    echo ""
    
    cmd_analyze
    echo ""
    
    cmd_select
    echo ""
    
    cmd_build
    echo ""
    
    cmd_test
    echo ""
    
    log_info "Smart build workflow complete! ✓"
}

cmd_stats() {
    log_info "Smart Build Statistics"
    echo ""
    
    cd "$BUILD_DIR" || exit 1
    
    if [ -f "enhanced_dependency_mapping.json" ]; then
        echo "Dependency Map:"
        local num_files=$(jq '.file_to_executables | length' enhanced_dependency_mapping.json)
        echo "  Total files tracked: $num_files"
        
        # Check core.hpp as example
        if jq -e '.file_to_executables["include/ck_tile/core.hpp"]' enhanced_dependency_mapping.json &> /dev/null; then
            local core_deps=$(jq '.file_to_executables["include/ck_tile/core.hpp"] | length' enhanced_dependency_mapping.json)
            echo "  Executables depending on core.hpp: $core_deps"
        fi
    else
        log_warn "Dependency map not found. Run 'analyze' first."
    fi
    
    echo ""
    
    if [ -f "tests_to_run.json" ]; then
        echo "Test Selection:"
        jq '.statistics' tests_to_run.json
        
        echo ""
        echo "Changed files:"
        jq -r '.changed_files[]' tests_to_run.json
        
        echo ""
        echo "Sample affected tests (first 10):"
        jq -r '.executables[:10][]' tests_to_run.json
    else
        log_warn "Test selection not found. Run 'select' first."
    fi
}

cmd_clean() {
    log_info "Cleaning generated smart build files..."
    
    cd "$BUILD_DIR" || exit 1
    
    rm -f enhanced_dependency_mapping.json tests_to_run.json build_mode.env
    
    log_info "Clean complete ✓"
}

# Parse command line arguments
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        analyze|select|build|test|all|stats|clean)
            COMMAND="$1"
            shift
            ;;
        -b|--base-ref)
            BASE_REF="$2"
            shift 2
            ;;
        -t|--target-ref)
            TARGET_REF="$2"
            shift 2
            ;;
        -j|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --no-ctest-only)
            CTEST_ONLY=""
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Validate command
if [ -z "$COMMAND" ]; then
    log_error "No command specified"
    print_help
    exit 1
fi

# Main execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_info "Script location: $SCRIPT_DIR"

cd "$SCRIPT_DIR" || exit 1

check_prerequisites

case "$COMMAND" in
    analyze)
        cmd_analyze
        ;;
    select)
        cmd_select
        ;;
    build)
        cmd_build
        ;;
    test)
        cmd_test
        ;;
    all)
        cmd_all
        ;;
    stats)
        cmd_stats
        ;;
    clean)
        cmd_clean
        ;;
esac

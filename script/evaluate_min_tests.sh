# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go one level up to PACKAGE_HOME
PACKAGE_HOME="$(dirname "$SCRIPT_DIR")"

# Search for build.ninja under PACKAGE_HOME
BUILD_NINJA_FILE=$(find "$PACKAGE_HOME" -name build.ninja -print -quit)

if [ -z "$BUILD_NINJA_FILE" ]; then
    echo "Error: build.ninja not found under $PACKAGE_HOME"
    exit 1
fi

python3 dependency-parser/main.py parse "$BUILD_NINJA_FILE" --workspace-root "$PACKAGE_HOME"

# Get the directory containing build.ninja
BUILD_DIR=$(dirname "$BUILD_NINJA_FILE")

# Path to enhanced_dependency_mapping.json in the same directory
JSON_FILE="$BUILD_DIR/enhanced_dependency_mapping.json"

# Check if the JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: $JSON_FILE not found."
    exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)

# Run the command
python3 dependency-parser/main.py select "$JSON_FILE" origin/amd-develop $branch
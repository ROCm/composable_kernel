#!/bin/bash
# my_script.sh

echo "Running my custom bash script..."
echo "Current directory: $(pwd)"
echo "Arguments received: $@"

# Example: Perform a check
if [ "$1" == "success" ]; then
    echo "Script finished successfully!"
    exit 0 # Indicate success
else
    echo "Script failed with argument: $1"
    exit 1 # Indicate failure
fi

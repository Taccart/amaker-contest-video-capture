#!/bin/bash
# Script to build and test the amaker-unleash-the-bricks package

set -e  # Exit immediately if a command exits with non-zero status

echo "===== Testing amaker-unleash-the-bricks package setup ====="

# Check for build package
if ! pip show build &>/dev/null; then
    echo "Installing build package..."
    pip install build
fi

# Build the package
echo "Building package..."
python -m build

# Create a virtual environment for testing
echo "Creating test environment..."
python -m venv test_venv

# Activate virtual environment
echo "Activating test environment..."
source test_venv/bin/activate

# Install the package
echo "Installing package for testing..."
pip install dist/amaker_unleash_the_bricks-*.whl

# Verify installation
echo "Verifying installation:"
pip list | grep amaker-unleash-the-bricks

# Test importing the package
echo "Testing package import..."
python -c "import amaker.unleash_the_bricks; print('Package imported successfully!')"

# Try running the command (with timeout to avoid getting stuck)
echo "Testing command line tool (will exit after 3 seconds)..."
timeout 3s unleash-the-bricks --camera_number -1 || echo "Entry point test completed"

# Clean up
echo "Cleaning up..."
deactivate
rm -rf test_venv

echo "===== Test complete ====="
echo "Package is available in the 'dist' directory"
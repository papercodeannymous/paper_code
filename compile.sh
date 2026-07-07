#!/bin/bash

# RTree C++ shared library compilation script
# compile C++ code in /rtree to cpp_out
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_FILE="${SCRIPT_DIR}/rtree/myrtree.cpp"
OUTPUT_DIR="${SCRIPT_DIR}/cpp_out"
OUTPUT_FILE="${OUTPUT_DIR}/libmytree.so"


# check vars name
# echo "SCRIPT_DIR: $SCRIPT_DIR"
# echo "SOURCE_FILE: $SOURCE_FILE"
# echo "OUTPUT_DIR: $OUTPUT_DIR"
# echo "OUTPUT_FILE: $OUTPUT_FILE"

# output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building RTree Shared Library ===${NC}"

# create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory..."
    mkdir -p $OUTPUT_DIR
fi

# check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}Error: Source file $SOURCE_FILE not found!${NC}"
    exit 1
fi

# compile the shared library
# PS: using -Wno-deprecated-copy to suppress warnings about deprecated copy constructors in C++11 code
echo "Compiling myrtree.cpp to shared library..."
g++ -shared -fPIC -o "$OUTPUT_FILE" \
    -std=c++11 \
    -Wall \
    -O2 \
    -Wno-sign-compare \
    -Wno-deprecated-copy \
    -Wno-unused-variable \
    -Wno-unused-but-set-variable \
    -Wno-unused-parameter \
    -Wno-maybe-uninitialized \
    "$SOURCE_FILE"

# check if build successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${YELLOW}Output: $OUTPUT_FILE${NC}"

    echo ""
    echo "Library information:"
    ls -lh "$OUTPUT_FILE"
    
    echo ""
    echo "Exported symbols (C interface):"
    nm -D "$OUTPUT_FILE" | grep " T " | grep -E "(Construct|Insert|Split|Query|Clear|Get|Set|Make|Action|Copy|Traverse|Retrieve)" | awk '{print "  " $3}'
    
    echo ""
    echo -e "${GREEN}=== Build Complete ===${NC}"
    echo ""
    echo "Usage in Python:"
    echo "  from ctypes import CDLL"
    echo "  lib = CDLL('${OUTPUT_FILE}')"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi
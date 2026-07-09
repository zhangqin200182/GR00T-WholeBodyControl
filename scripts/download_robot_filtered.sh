#!/bin/bash
# Download robot_filtered (G1 retargeted motions) from BONES-SEED on HuggingFace.
# Usage: bash scripts/download_robot_filtered.sh
set -e

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
TARBALL="$DATA_DIR/g1.tar.gz"
EXTRACT_DIR="$DATA_DIR/robot_filtered"

mkdir -p "$DATA_DIR"

if [ -f "$TARBALL" ]; then
    echo "=== Tarball exists: $(ls -lh "$TARBALL" | awk '{print $5}') ==="
else
    echo "=== Downloading g1.tar.gz (21.9 GB) from HuggingFace ==="
    echo "This may take 1-3 hours depending on network speed."
    echo ""

    # Use hf download with resume support
    hf download bones-studio/seed \
        --repo-type dataset \
        g1.tar.gz \
        --local-dir "$DATA_DIR"

    echo ""
    echo "=== Download complete: $(ls -lh "$TARBALL" | awk '{print $5}') ==="
fi

# Extract
if [ -d "$EXTRACT_DIR" ] && [ "$(ls "$EXTRACT_DIR"/*.pkl 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "=== Already extracted: $(ls "$EXTRACT_DIR"/*.pkl 2>/dev/null | wc -l) PKL files ==="
else
    echo "=== Extracting g1.tar.gz ... ==="
    mkdir -p "$EXTRACT_DIR"
    tar xf "$TARBALL" -C "$EXTRACT_DIR" --strip-components=1
    echo "=== Extraction complete: $(ls "$EXTRACT_DIR"/*.pkl 2>/dev/null | wc -l) PKL files ==="
fi

echo ""
echo "=== Done. Data at: $EXTRACT_DIR ==="
du -sh "$EXTRACT_DIR"

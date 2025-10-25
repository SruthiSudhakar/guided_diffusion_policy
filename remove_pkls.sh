#!/bin/bash
# Remove all .pkl files inside the target directory and its subfolders

TARGET_DIR="/app/data/checkpoints/dp_model/epoch=1100-val_loss=0.037"

echo "Deleting all .pkl files under: $TARGET_DIR"

# Safety check: make sure directory exists before running rm
if [ -d "$TARGET_DIR" ]; then
    find "$TARGET_DIR" -type f -name "*.pkl" -print -delete
    echo "✅ All .pkl files removed."
else
    echo "❌ Directory not found: $TARGET_DIR"
fi

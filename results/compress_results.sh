#!/bin/bash

# Base directory containing the results
BASE_DIR="./results"

# Function to compress a subdirectory
compress_directory() {
    local dir="$1"
    local parent_dir
    parent_dir=$(dirname "$dir")
    local archive_name="$dir.tar.gz"

    echo "Compressing '$dir' into '$archive_name'..."
    tar -czf "$archive_name" -C "$parent_dir" "$(basename "$dir")"
    if [ $? -eq 0 ]; then
        echo "Successfully compressed '$dir'."

        # Add the directory to .gitignore if not already present
        if ! grep -qx "$dir/" .gitignore; then
            echo "$dir/" >> .gitignore
            echo "Added '$dir/' to .gitignore."
        else
            echo "'$dir/' is already in .gitignore, skipping addition."
        fi
    else
        echo "Failed to compress '$dir'."
    fi
}

# Process all subdirectories inside the results subdirectories
process_results_subdirectories() {
    for subdir in "$BASE_DIR"/*/*; do
        if [ -d "$subdir" ] && [[ "$subdir" != *"/local/"* ]] ; then
            compress_directory "$subdir"
        fi
    done
}

# Main execution
process_results_subdirectories

# Deduplicate and sort .gitignore
sort -u .gitignore -o .gitignore
echo ".gitignore cleaned and sorted."

# Stage .gitignore
git add .gitignore
echo ".gitignore staged for commit."

#!/bin/bash

# File containing the list of URLs
URL_LIST="data/files_to_download.txt"

# Directory to save the images (mirrors PhysioNet structure)
DOWNLOAD_DIR="data/mimic-cxr-jpg-2.0.0/files"

# Initial Checks
echo "--- Starting PhysioNet Image Download ---"
echo "Reading URLs from: $URL_LIST"
echo "Downloading images to: $DOWNLOAD_DIR"

# Ensure base download directory exists
mkdir -p "$DOWNLOAD_DIR"

# Check if URL list exists
if [ ! -f "$URL_LIST" ]; then
    echo "❌ Error: $URL_LIST not found. Please run the Python script to generate it first."
    exit 1
fi

# Credentials
USERNAME="bharath1675"
echo -n "Please enter your PhysioNet password: "
read -s PASSWORD
echo ""

# Download Files
while IFS= read -r url; do
    # Skip empty lines
    if [ -z "$url" ]; then
        continue
    fi

    # Extract relative path (remove the base URL)
    relative_path=$(echo "$url" | sed "s|https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/||")

    # Full path where file will be stored
    local_file="$DOWNLOAD_DIR/$relative_path"

    # Create subdirectories if needed
    mkdir -p "$(dirname "$local_file")"

    # If file already exists and is complete, skip
    if [ -f "$local_file" ]; then
        echo "✔ Skipping (already exists): $relative_path"
        continue
    fi

    # Download with authentication
    echo "⬇ Downloading: $relative_path"
    wget --user="$USERNAME" --password="$PASSWORD" \
         --continue \
         --no-host-directories --cut-dirs=4 \
         -O "$local_file" "$url"

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "⚠️ Failed to download: $url"
    fi
done < "$URL_LIST"

echo "🎉 Download complete! Files saved in: $DOWNLOAD_DIR"

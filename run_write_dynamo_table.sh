#!/bin/bash

# File containing the list of tomograms
TOMO_LIST="/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS/tomo_for_sta.txt"

# Base directory for tomograms
BASE_DIR="/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS/"

# Initialize ID counter
ID=1

# Read each tomogram from the list
while read -r tomo; do
    # Construct the directory path
    dir="$BASE_DIR/$tomo/best_alignment"
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir with ID: $ID"
        # Navigate to the directory and run the command
        (cd "$dir" && finding_ampa write-dynamo-table --tomo-id "$ID" ampa_poses/ampa_pose_1.star )
        # Increment the ID counter
        ID=$((ID + 1))
    else
        echo "Directory does not exist: $dir"
        ID=$((ID + 1))
    fi
done < "$TOMO_LIST"

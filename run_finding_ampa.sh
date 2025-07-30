#!/bin/bash

# File containing the list of tomograms
TOMO_LIST="/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS/tomo_for_sta.txt"

# Base directory for tomograms
BASE_DIR="/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS/"

# Read each tomogram from the list
while read -r tomo; do
    # Construct the directory path
    dir="$BASE_DIR/$tomo/best_alignment"
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        # Navigate to the directory and run the command
        (cd "$dir" && finding_ampa create-relion-starfile ampa_pose_2 --aunp-distance-cutoffs-a 30 150 --membrane-distance-cutoffs-a 100 300)
    else
        echo "Directory does not exist: $dir"
    fi
done < "$TOMO_LIST"

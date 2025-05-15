#!/bin/bash

FASTQC="./FastQC/fastqc"
DATA_DIR="./../data_acquisition/datatest"

# Loop over each folder inside the data directory
for FOLDER in "$DATA_DIR"/*; do
    if [[ -d "$FOLDER" ]]; then
        echo "Searching for .fastq files in $FOLDER..."

        # Find all .fastq files in the current folder
        FASTQ_FILES=("$FOLDER"/*.fastq)

        # Check if any .fastq files were found
        if [[ -e "${FASTQ_FILES[0]}" ]]; then
            echo "Running FastQC on files in $FOLDER..."
            "$FASTQC" --extract "${FASTQ_FILES[@]}" --outdir=./fastqc-result
            echo "Finished FastQC for $FOLDER"
        else
            echo "No .fastq files found in $FOLDER. Skipping..."
        fi
    fi
done

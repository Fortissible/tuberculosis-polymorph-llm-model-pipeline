#!/bin/bash
Dir="./datatest/"
cd "$Dir"
for Folder in *; do
    if [ -f "${Folder}/${Folder}_2.fastq.gz" ]; then
        gunzip -v "${Folder}/${Folder}_1.fastq.gz" "${Folder}/${Folder}_2.fastq.gz"
        echo "${Folder} selesai di-extract."
    else
        echo "${Folder}_2.fastq sudah di-compress atau file tidak ditemukan."
    fi
done

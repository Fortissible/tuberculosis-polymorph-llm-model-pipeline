#!/bin/bash

BOWTIE="./bowtie2/bowtie2"
SEQ1="./trimmomatic-result/ERR2512812-trimmed/ERR2512812_forward_paired.fq"
SEQ2="./trimmomatic-result/ERR2512812-trimmed/ERR2512812_reverse_paired.fq"

"${BOWTIE}" -x mtb-refseq-H37Rv -1 "${SEQ1}" -2 "${SEQ2}" -S aligned.sam

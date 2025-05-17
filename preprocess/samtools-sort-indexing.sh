#!/bin/bash

SAMTOOLS="./samtools/samtools"
SAM_INPUT="aligned"

"${SAMTOOLS}" view -Sb "${SAM_INPUT}".sam | "${SAMTOOLS}" sort -o "${SAM_INPUT}".sorted.bam && "${SAMTOOLS}" index "${SAM_INPUT}".sorted.bam





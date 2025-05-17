#!/bin/bash

# INPUTS
BAM="aligned.sorted.bam"
PREFIX="ERR2512812"
REF="mtb-refseq-H37Rv.fasta"

# Required Tools
BCFTOOLS="./bcftools/bcftools"

# 1. Variant calling
$BCFTOOLS mpileup -Ou -f $REF $BAM | \
$BCFTOOLS call -mv -Ov -o ${PREFIX}.raw.vcf

# 2. Filter VCF: depth ≥ 10, phred ≥ 30
$BCFTOOLS filter -i 'DP>=10 && QUAL>=30' ${PREFIX}.raw.vcf > ${PREFIX}.filtered.vcf
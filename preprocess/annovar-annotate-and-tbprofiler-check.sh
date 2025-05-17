#!/bin/bash

# INPUTS
BAM="aligned.sorted.bam"
PREFIX="ERR2512812"
ANNOVAR_DB="./annovar/humandb/"
BUILDVER="hg19"

# REQUIRED FUNCTION & TOOLS
CONVERT2ANNOVAR="./annovar/convert2annovar.pl"
TABLEANNOVAR="./annovar/table_annovar.pl"

# 1. Convert VCF to ANNOVAR input
$CONVERT2ANNOVAR -format vcf4 ${PREFIX}.filtered.vcf > ${PREFIX}.avinput

# 2. Annotate with ANNOVAR (missense, synonymous, etc.)
$TABLEANNOVAR ${PREFIX}.avinput $ANNOVAR_DB \
-buildver $BUILDVER \
-out ${PREFIX}_annotated \
-remove -protocol refGene \
-operation g -nastring . -csvout -polish

# 3. TB-Profiler for drug resistance classification
tb-profiler profile -a $BAM -p ${PREFIX}_tbprofiler
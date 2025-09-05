#!/bin/bash

INPUT_FILE="ERR2512812.renamed.vcf"
ANN_FILE="ERR2512812.renamed.ann.vcf"
#ANN_PASS_FILE="ERR2512812.renamed.ann.pass.vcf"
OUTPUT_FILE="ERR2512812.renamed.tsv"

# 0) annotate with snpEff (add -hgvs so HGVS_C/HGVS_P are filled)
java -Xmx8g -jar snpEff/snpEff.jar -hgvs Mycobacterium_tuberculosis_h37rv \
  $INPUT_FILE -v > $ANN_FILE

# 1) filter with SnpSift → keep only PASS & QUAL>=30  (strict version)
#java -Xmx2g -jar snpEff/SnpSift.jar filter \
#  "( FILTER = 'PASS' ) & ( QUAL >= 30 )" \
#  $ANN_FILE -v > $ANN_PASS_FILE

# --- OR (recommended if your FILTER is often '.') ---
# java -Xmx2g -jar snpEff/SnpSift.jar filter \
#   "( ( FILTER = 'PASS' ) | ( ! exists FILTER ) | ( FILTER = '.' ) ) & ( QUAL >= 30 ) & ( exists ANN )" \
#   $ANN_FILE > $ANN_PASS_FILE

# 2) extract ANN fields into a TSV
java -Xmx2g -jar snpEff/SnpSift.jar extractFields -s "," \
  $ANN_FILE \
  CHROM POS REF ALT -v \
  "ANN[*].EFFECT" "ANN[*].IMPACT" "ANN[*].GENE" "ANN[*].GENEID" \
  "ANN[*].HGVS_C" "ANN[*].HGVS_P" "ANN[*].FEATURE" "ANN[*].FEATUREID" "ANN[*].BIOTYPE" \
  > $OUTPUT_FILE


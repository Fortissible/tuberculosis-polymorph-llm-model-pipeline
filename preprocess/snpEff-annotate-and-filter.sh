#!/bin/bash

# Input folder from the Python script (same level as this .sh)
INPUT_DIR="./dataset-test/contigs-renamed"
# Output folder for annotated VCFs and TSVs
OUTPUT_DIR="./dataset-test/snpeff-out"

for INPUT_FILE in "$INPUT_DIR"/*.vcf; do
  bn="$(basename "$INPUT_FILE" .vcf)"
  ANN_FILE="${OUTPUT_DIR}/${bn}.ann.vcf"
  OUTPUT_FILE="${OUTPUT_DIR}/${bn}.tsv"

  # 0) snpEff annotate (fills HGVS_C/HGVS_P)
  java -Xmx8g -jar snpEff/snpEff.jar -hgvs Mycobacterium_tuberculosis_h37rv \
    "$INPUT_FILE" -v > "$ANN_FILE"

  # 2) extract ANN fields into a TSV
  java -Xmx2g -jar snpEff/SnpSift.jar extractFields -s "," \
    "$ANN_FILE" \
    CHROM POS REF ALT -v \
    "ANN[*].EFFECT" "ANN[*].IMPACT" "ANN[*].GENE" "ANN[*].GENEID" \
    "ANN[*].HGVS_C" "ANN[*].HGVS_P" "ANN[*].FEATURE" "ANN[*].FEATUREID" "ANN[*].BIOTYPE" \
    > "$OUTPUT_FILE"

  echo "Done: $INPUT_FILE -> $ANN_FILE, $OUTPUT_FILE"
done


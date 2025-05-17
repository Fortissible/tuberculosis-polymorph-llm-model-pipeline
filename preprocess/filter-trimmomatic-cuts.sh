#!/bin/bash

TRIMMOMATIC="./trimmomatic/bin/trimmomatic.jar"
FASTQC_RESULTS="./fastqc-result"
DATA_DIR_FW="./../data_acquisition/datatest/ERR2512812/ERR2512812_1.fastq"
DATA_DIR_BW="./../data_acquisition/datatest/ERR2512812/ERR2512812_2.fastq"
# ILLUMINA_ADAPTER="./trimmomatic/adapters/TruSeq3-PE.fa"

# add this arg to the trimmomatic call for cutting the adapters
# ILLUMINACLIP:"${ILLUMINA_ADAPTER}":2:30:10:2:True \

# modify the args based on the ${FASTQC_RESULTS}
java -jar "${TRIMMOMATIC}" PE \
  "${DATA_DIR_FW}" "${DATA_DIR_BW}" \
  ERR2512812_forward_paired.fq.gz ERR2512812_forward_unpaired.fq.gz \
  ERR2512812_reverse_paired.fq.gz ERR2512812_reverse_unpaired.fq.gz \
  LEADING:3 TRAILING:3 MINLEN:36
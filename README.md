# tuberculosis-polymorph-llm-model-pipeline
 This project is a bioinformatics tool & python based pipeline for Modeling the Association of Gene Polymorphisms and Drug Resistance in Mycobacterium tuberculosis Using Deep Learning and Large Language Models
 
 ```use dos2unix if there's error when running the sh script```
 
# Tools
Tools/library used in the pipeline :
- enaWebTools : https://github.com/enasequence/enaBrowserTools
- fastqc : https://www.bioinformatics.babraham.ac.uk/projects/fastqc/
- trimmomatic : https://github.com/timflutre/trimmomatic
- cutadapt : https://cutadapt.readthedocs.io/en/stable/guide.html#basic-usage
- blast+ : https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html
- blast+ required dependencies (sqlite): https://www.sqlite.org/download.html
- kraken-tools : https://github.com/jenniferlu717/KrakenTools
- bowtie2 : https://github.com/BenLangmead/bowtie2
- samtools & bcftools : https://www.htslib.org/
- annovar : https://annovar.openbioinformatics.org/en/latest/user-guide/startup/
- miniconda : https://www.anaconda.com/docs/getting-started/miniconda/install
- tb-profiler : https://github.com/jodyphelan/TBProfiler
- scikit-learn : https://github.com/scikit-learn/scikit-learn
- tensorflow : https://pypi.org/project/tensorflow/
- pytorch : https://pypi.org/project/torch/
- Other library needed in python3 is Numpy,Pandas,sklearn,matplotlib, etc.

using blast+ with downloaded db for specify non-target contaminant
```
blastn -query unknown_sequences.fasta -db nt -out results.out -outfmt 6 -max_target_seqs 5 -evalue 1e-5
```

# Flowchart
<p align="center">
<img src="/img/llm-mtb-polymorp-pipeline.drawio.png" width="480" title="pipeline">
</p>

# Step-By-Step
- fill the ```/data_acquisition/accession.txt``` with accession number list 
  (*separated with newline/enter*)
- run ```bash /data_acquisition/download_accessions_datatest.sh``` to download the sequence 
  to folder ```/data_acquisition/datatest```
- run ```bash /data_acquisition/gzip_decompress_fastq.sh``` to decompress the gzip file <br> or 
  ```bash /data_acquisition/gzip_compress_fastq.sh``` to compress fastq to gzip
- after sequence download complete, then continue to download and install the required deps (*read the installation manual from the url*):
  - fastqc : https://www.bioinformatics.babraham.ac.uk/projects/fastqc/
  - trimmomatic : https://github.com/timflutre/trimmomatic
  - bowtie2 : https://github.com/BenLangmead/bowtie2
  - samtools & bcftools : https://www.htslib.org/
  - annovar : https://annovar.openbioinformatics.org/en/latest/user-guide/download/
  - miniconda : https://www.anaconda.com/docs/getting-started/miniconda/install
  - tbprofiler : https://github.com/jodyphelan/TBProfiler
  
- make sure the required deps installation location is in :
  - ```/preprocess/bcftools```
  - ```/preprocess/bowtie2```
  - ```/preprocess/FastQC```
  - ```/preprocess/samtools```
  - ```/preprocess/trimmomatic```
  - ```/preprocess/annovar```
  
- run ```bash extract-fastqc-analysis.sh``` to extract sequence detail and analysis from the downloaded data in datatest to ```/preprocess/fastqc-result```
- run ```bash filter-trimmomatic-cuts.sh``` for filter low-qc sequence based on the threshold of fastqc analysis to the folder ```/preprocess/trimmomatic-result```
- run ```bash alignment-bowtie-refseq-build-idx.sh``` to index the reference sequence ```/preprocess/mtb-refseq-H37Rv.fasta```
- run ```bash alignment-bowtie.sh``` to do the alignment of the filtered data sequence in ```/preprocess/trimmomatic-result``` with indexed reference sequence
- run ```bash samtools-sort-indexing.sh``` for index and sort the output of bowtie alignment with reference sequence the output is ```.bam``` and ```.bam.bai``` files
- run ```bash bcftools-varcall-filter.sh``` to variant call and filter the aligned output .bam and index .bam files
- run ```bash annovar-annotate-and-tbprofiler-check.sh``` to annotate the result using annovar and crosscheck the mutation and SNP to the TBProfiler DB.

if you want deinit conda script, run with ```conda init --reverse $SHELL```

# What to do next 
- Preprocess
- SNP & Feature extraction
- LLM & DL Modelling

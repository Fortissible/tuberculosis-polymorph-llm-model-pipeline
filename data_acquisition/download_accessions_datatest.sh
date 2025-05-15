#!/bin/bash
File="accession.txt"
Lines=$(cat $File)
for Line in $Lines
do
	python3 ./enaBrowserTools-master/python3/enaDataGet.py -f fastq -d ./datatest $Line
done
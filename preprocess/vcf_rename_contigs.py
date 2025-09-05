# vcf_rename_contig.py
import sys
old, new = "NC_000962.3", "Chromosome"
with open(sys.argv[1], "r") as fin, open(sys.argv[2], "w") as fout:
    for line in fin:
        if line.startswith("##contig=<ID=" + old):
            line = line.replace("##contig=<ID=" + old, "##contig=<ID=" + new)
        elif not line.startswith("#"):
            # CHROM column is first field
            parts = line.split("\t")
            if parts[0] == old:
                parts[0] = new
                line = "\t".join(parts)
        fout.write(line)

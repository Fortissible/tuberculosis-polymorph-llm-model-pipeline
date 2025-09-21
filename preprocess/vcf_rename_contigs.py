#!/usr/bin/env python3
import os
import sys

# Change these if needed
OLD = "NC_000962.3"
NEW = "Chromosome"

def process_vcf(in_path: str, out_path: str, old: str = OLD, new: str = NEW) -> None:
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("##contig=<ID=" + old):
                line = line.replace("##contig=<ID=" + old, "##contig=<ID=" + new)
            elif not line.startswith("#"):
                parts = line.split("\t")
                if parts and parts[0] == old:
                    parts[0] = new
                    line = "\t".join(parts)
            fout.write(line)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.isdir(in_dir):
        print(f"Error: input_dir not found or not a directory: {in_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Process only top-level .vcf files (non-recursive as requested)
    vcf_files = [f for f in os.listdir(in_dir) if f.lower().endswith(".vcf")]

    if not vcf_files:
        print(f"No .vcf files found in {in_dir}", file=sys.stderr)
        sys.exit(0)

    for name in vcf_files:
        in_path = os.path.join(in_dir, name)
        base, _ = os.path.splitext(name)  # remove .vcf
        out_name = f"{base}-rename-contigs.vcf"
        out_path = os.path.join(out_dir, out_name)

        process_vcf(in_path, out_path)
        print(f"Processed: {in_path} -> {out_path}")

if __name__ == "__main__":
    main()

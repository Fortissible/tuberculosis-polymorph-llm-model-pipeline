# build_mutkeys.py
import pandas as pd
import pathlib, re, sys, argparse

EFFECT_PRIORITY = [
    "stop_gained", "stop_lost", "start_lost", "frameshift_variant",
    "missense_variant",
    "synonymous_variant",
    "splice_acceptor_variant", "splice_donor_variant", "splice_region_variant",
    "upstream_gene_variant", "downstream_gene_variant", "regulatory_region_variant",
    "intergenic_region"
]

def best_ann(effects):
    for eff in EFFECT_PRIORITY:
        for i, e in enumerate(effects):
            if eff in e:
                return i
    return 0 if effects else None

def parse_csv(tsv_path: pathlib.Path):
    required = {"CHROM","POS","REF","ALT",
                "ANN[*].EFFECT","ANN[*].GENE","ANN[*].HGVS_C","ANN[*].HGVS_P"}
    try:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    except Exception as e:
        print(f"[WARN] Failed to read {tsv_path.name}: {e}", file=sys.stderr)
        return []

    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] {tsv_path.name} missing columns: {sorted(missing)} — skipping", file=sys.stderr)
        return []

    colmap = {
        "ANN[*].EFFECT":"EFFECTS","ANN[*].IMPACT":"IMPACTS","ANN[*].GENE":"GENES","ANN[*].GENEID":"GENEIDS",
        "ANN[*].HGVS_C":"HGVS_C","ANN[*].HGVS_P":"HGVS_P","ANN[*].FEATURE":"FEATURES",
        "ANN[*].FEATUREID":"FEATUREIDS","ANN[*].BIOTYPE":"BIOTYPES"
    }
    df = df.rename(columns={k:v for k,v in colmap.items() if k in df.columns})

    mutkeys = []
    for _, r in df.iterrows():
        effects  = [x.strip() for x in r.get("EFFECTS","").split(",") if x!=""]
        genes    = [x.strip() for x in r.get("GENES","").split(",")   if x!=""]
        hgvs_c   = [x.strip() for x in r.get("HGVS_C","").split(",")  if x!=""]
        hgvs_p   = [x.strip() for x in r.get("HGVS_P","").split(",")  if x!=""]

        L = max(len(effects), len(genes), len(hgvs_c), len(hgvs_p), 1)
        while len(effects) < L: effects.append("")
        while len(genes)   < L: genes.append("")
        while len(hgvs_c)  < L: hgvs_c.append("")
        while len(hgvs_p)  < L: hgvs_p.append("")

        idx = best_ann(effects)
        if idx is None:
            continue

        eff  = effects[idx]
        gene = genes[idx] or "intergenic"
        c    = hgvs_c[idx].replace("c.","") if hgvs_c[idx] else ""
        p    = hgvs_p[idx].replace("p.","") if hgvs_p[idx] else ""

        chrom, pos, ref, alt = r["CHROM"], r["POS"], r["REF"], r["ALT"]
        refalt = f"{pos}{ref}>{alt}"

        if any(x in eff for x in ["missense_variant","stop_gained","stop_lost","start_lost","frameshift_variant"]):
            key = f"{gene}|p.{p}" if p else f"{gene}|{refalt}"
        elif "synonymous_variant" in eff:
            key = f"{gene}|syn:{c}" if c else f"{gene}|syn:{refalt}"
        elif any(x in eff for x in ["upstream_gene_variant","downstream_gene_variant","regulatory_region_variant"]):
            key = f"{gene}|promoter:{refalt}"
        elif "intergenic_region" in eff or gene == "intergenic":
            key = f"intergenic|{chrom}:{refalt}"
        else:
            key = f"{gene}|{eff}:{refalt}"

        mutkeys.append(key)

    return sorted(set(mutkeys))

def clean_sample_id(stem: str) -> str:
    s = stem
    s = re.sub(r"\.ann(\.pass)?$", "", s)
    s = re.sub(r"\.renamed$", "", s)
    s = re.sub(r"-rename-contigs$", "", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="directory containing TSVs")
    ap.add_argument("--pattern", default="*.tsv", help="glob for TSV files (default: *.tsv)")
    ap.add_argument("--out_name", default="all_samples_mutkeys.csv",
                    help="output CSV file name (default: all_samples_mutkeys.csv)")
    ap.add_argument("--out_dir", default=None,
                    help="directory to write the output CSV (default: same as --in_dir)")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Error: input directory not found: {in_dir}")

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    rows = []
    tsvs = sorted(in_dir.glob(args.pattern))
    if not tsvs:
        print(f"[INFO] No TSVs matched {args.pattern} in {in_dir}")
    for tsv in tsvs:
        sample_id = clean_sample_id(tsv.stem)
        mk = parse_csv(tsv)
        rows.extend({"sample_id": sample_id, "mut_key": k} for k in mk)
        print(f"[OK] {tsv.name}: {len(mk)} unique mut_keys")

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[DONE] Wrote {out_path} with {len(rows)} rows.")
    else:
        print("[INFO] No rows to write (no mut_keys found).")

if __name__ == "__main__":
    main()

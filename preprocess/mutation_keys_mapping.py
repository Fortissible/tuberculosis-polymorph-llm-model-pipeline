# build_mutkeys.py
import pandas as pd, pathlib, re

# priority order for picking one ANN per ALT (tune as needed)
EFFECT_PRIORITY = [
    "stop_gained", "stop_lost", "start_lost", "frameshift_variant",
    "missense_variant",
    "synonymous_variant",
    "splice_acceptor_variant", "splice_donor_variant", "splice_region_variant",
    "upstream_gene_variant", "downstream_gene_variant", "regulatory_region_variant",
    "intergenic_region"
]

def best_ann(effects):
    # choose the first effect that appears in priority list
    for eff in EFFECT_PRIORITY:
        for i, e in enumerate(effects):
            if eff in e:
                return i
    return 0 if effects else None

def parse_csv(tsv_path: pathlib.Path):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    # rename for convenience (no brackets)
    colmap = {
        "ANN[*].EFFECT":"EFFECTS","ANN[*].IMPACT":"IMPACTS","ANN[*].GENE":"GENES","ANN[*].GENEID":"GENEIDS",
        "ANN[*].HGVS_C":"HGVS_C","ANN[*].HGVS_P":"HGVS_P","ANN[*].FEATURE":"FEATURES",
        "ANN[*].FEATUREID":"FEATUREIDS","ANN[*].BIOTYPE":"BIOTYPES"
    }
    df = df.rename(columns=colmap)
    mutkeys = []

    for _, r in df.iterrows():
        # split parallel lists
        effects  = [x.strip() for x in r.get("EFFECTS","").split(",") if x!=""]
        genes    = [x.strip() for x in r.get("GENES","").split(",")   if x!=""]
        hgvs_c   = [x.strip() for x in r.get("HGVS_C","").split(",")  if x!=""]
        hgvs_p   = [x.strip() for x in r.get("HGVS_P","").split(",")  if x!=""]
        # fallback if lists are shorter; pad
        L = max(len(effects), len(genes), len(hgvs_c), len(hgvs_p), 1)
        while len(effects) < L: effects.append("")
        while len(genes)   < L: genes.append("")
        while len(hgvs_c)  < L: hgvs_c.append("")
        while len(hgvs_p)  < L: hgvs_p.append("")

        idx = best_ann(effects)
        if idx is None:
            continue
        eff = effects[idx]
        gene = genes[idx] or "intergenic"
        c = hgvs_c[idx].replace("c.","") if hgvs_c[idx] else ""
        p = hgvs_p[idx].replace("p.","") if hgvs_p[idx] else ""

        chrom, pos, ref, alt = r["CHROM"], r["POS"], r["REF"], r["ALT"]
        refalt = f"{pos}{ref}>{alt}"

        # mut_key rules
        if any(x in eff for x in ["missense_variant","stop_gained","stop_lost","start_lost","frameshift_variant"]):
            key = f"{gene}|p.{p}" if p else f"{gene}|{refalt}"
        elif "synonymous_variant" in eff:
            key = f"{gene}|syn:{c}" if c else f"{gene}|syn:{refalt}"
        elif any(x in eff for x in ["upstream_gene_variant","downstream_gene_variant","regulatory_region_variant"]):
            key = f"{gene}|promoter:{refalt}"
        elif "intergenic_region" in eff or gene == "intergenic":
            key = f"intergenic|{chrom}:{refalt}"
        else:
            # generic fallback with effect label
            key = f"{gene}|{eff}:{refalt}"
        mutkeys.append(key)

    return sorted(set(mutkeys))

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="folder with per-sample TSVs")
    ap.add_argument("--pattern", default="*.tsv", help="glob for TSV's files")
    ap.add_argument("--out_pairs", default="all_samples_mutkeys.csv", help="sample_id,mut_key rows")
    args = ap.parse_args()

    rows = []
    for tsv in pathlib.Path(args.in_dir).glob(args.pattern):
        sample_id = tsv.stem  # e.g., ERR2512812.ann.pass -> adjust if needed
        # strip common suffixes
        sample_id = re.sub(r"\.ann(\.pass)?$", "", sample_id)
        mk = parse_csv(tsv)
        rows.extend([{"sample_id": sample_id, "mut_key": k} for k in mk])

    pd.DataFrame(rows).to_csv(args.out_pairs, index=False)
    print(f"Wrote {args.out_pairs} with {len(rows)} rows.")

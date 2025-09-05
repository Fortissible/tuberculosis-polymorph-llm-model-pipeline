# make_labels.py
import pandas as pd

# Load features
X = pd.read_csv("variant_presence_matrix.csv", index_col=0)

# Load meta
meta = pd.read_csv("alldata_sheet16_classes.csv")

# Normalize IDs so they match the feature matrix index.
# If your build_mutkeys.py used filename stems that match 'accession', this will align.
meta = meta.rename(columns={"accession": "sample_id"}).set_index("sample_id")

# Helper to map string phenotype to int if numeric not present
def map_str(col):
    return col.map({"R": 1, "S": 0}).astype("Int64")

# Build label columns (prefer numeric phen_*; else map from string)
labels = pd.DataFrame(index=meta.index)
labels["label_INH"] = meta.get("phen_inh")
if labels["label_INH"].isnull().any() and "Phenotype_Isoniazid" in meta:
    labels["label_INH"] = labels["label_INH"].fillna(map_str(meta["Phenotype_Isoniazid"]))
labels["label_RIF"] = meta.get("phen_rif")
if labels["label_RIF"].isnull().any() and "Phenotype_Rifampicin" in meta:
    labels["label_RIF"] = labels["label_RIF"].fillna(map_str(meta["Phenotype_Rifampicin"]))
labels["label_EMB"] = meta.get("phen_emb")
if labels["label_EMB"].isnull().any() and "Phenotype_Ethambutol" in meta:
    labels["label_EMB"] = labels["label_EMB"].fillna(map_str(meta["Phenotype_Ethambutol"]))
labels["label_PZA"] = meta.get("phen_pza")
if labels["label_PZA"].isnull().any() and "Phenotype_Pyrazinamide" in meta:
    labels["label_PZA"] = labels["label_PZA"].fillna(map_str(meta["Phenotype_Pyrazinamide"]))

# Lineage (as string); you can keep numeric values, we’ll one-hot later
lineage = meta["lineage"].astype(str)

# Keep only samples present in X, and drop rows with missing labels per drug later
labels = labels.loc[X.index.intersection(labels.index)]
lineage = lineage.loc[labels.index]

# Save a clean meta for training
clean = pd.concat([lineage.rename("lineage"), labels], axis=1)
clean.to_csv("meta_labels.csv")
print("Wrote meta_labels.csv with shape", clean.shape)

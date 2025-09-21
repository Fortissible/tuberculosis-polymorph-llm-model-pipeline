# build_matrix.py
import pandas as pd

pairs = pd.read_csv("E:/Project/tuberculosis-polymorph-llm-model-pipeline/preprocess/dataset-test/mutation-tables/all_samples_mutkeys.csv")  # columns: sample_id, mut_key
# optional: restrict to a curated list first (e.g., WHO catalogue mut_keys)
# cat = pd.read_csv("who_catalog_mutkeys.csv")  # columns: mut_key, drug, category
# pairs = pairs.merge(cat[["mut_key"]].drop_duplicates(), on="mut_key", how="inner")

X = (pairs.assign(val=1)
          .pivot_table(index="sample_id", columns="mut_key", values="val", fill_value=0)
          .astype("int8"))
X.to_csv("E:/Project/tuberculosis-polymorph-llm-model-pipeline/preprocess/dataset-test/mutation-tables/variant_presence_matrix.csv")
print("Matrix shape:", X.shape)

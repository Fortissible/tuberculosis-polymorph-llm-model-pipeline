# join_lineage.py
import pandas as pd

X = pd.read_csv("E:/Project/tuberculosis-polymorph-llm-model-pipeline/preprocess/dataset-test/mutation-tables/variant_presence_matrix-filtered.csv", index_col=0)
meta = pd.read_csv("E:/Project/tuberculosis-polymorph-llm-model-pipeline/preprocess/dataset-test/mutation-tables/meta_labels.csv", index_col=0)

# Align
idx = X.index.intersection(meta.index)
X = X.loc[idx]
meta = meta.loc[idx]

# One-hot lineage
lin = pd.get_dummies(meta["lineage"].astype(str), prefix="lin")
X_lin = X.join(lin)

X_lin.to_csv("E:/Project/tuberculosis-polymorph-llm-model-pipeline/preprocess/dataset-test/mutation-tables/features_with_lineage.csv")
print("features_with_lineage.csv", X_lin.shape)

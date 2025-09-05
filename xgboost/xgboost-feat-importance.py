import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

# Generate synthetic complex data
n_samples = 5000
df = pd.DataFrame({
    "age": np.random.normal(40, 12, size=n_samples),
    "income": np.random.normal(60000, 15000, size=n_samples),
    "job_type": np.random.choice(['tech', 'sales', 'admin', 'blue_collar'], size=n_samples),
    "marital_status": np.random.choice(['single', 'married', 'divorced'], size=n_samples),
    "num_children": np.random.poisson(1.5, size=n_samples),
    "owns_house": np.random.choice([0, 1], size=n_samples),
    "credit_score": np.random.normal(650, 50, size=n_samples),
    "loan_amount": np.random.normal(15000, 7000, size=n_samples)
})

# Inject nonlinear and interaction effects
df['credit_utilization'] = df['loan_amount'] / (df['income'] + 1)
df['income_log'] = np.log(df['income'] + 1)
df['age_squared'] = df['age'] ** 2

# Add some missing values
for col in ['income', 'credit_score']:
    df.loc[df.sample(frac=0.1).index, col] = np.nan

# Target variable (binary classification)
df['default'] = (
    (df['credit_score'] < 600) &
    (df['loan_amount'] > 20000) &
    (df['income'] < 50000)
).astype(int)

# =======================
# 🔧 Preprocessing
# =======================

# Impute missing values
df['income'].fillna(df['income'].median(), inplace=True)
df['credit_score'].fillna(df['credit_score'].median(), inplace=True)

# One-hot encode categorical variables
cat_cols = ['job_type', 'marital_status']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Separate features and target
X = df_encoded.drop('default', axis=1)
y = df_encoded['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 🧪 Train XGBoost Model
# =======================
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =======================
# 📊 Evaluation
# =======================
y_pred = model.predict(X_test_scaled)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =======================
# 📈 Feature Importance
# =======================
plt.figure(figsize=(12, 6))
plot_importance(model, max_num_features=15)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Configuration ---
# These are the files we created in Phase 1
X_TRAIN_FILE = "X_train.csv"

# --- These are the 3 NEW files we will CREATE ---
SCALER_CUSTOM_FILE = "scaler_custom.joblib"
SCALER_EMBED_FILE = "scaler_embed.joblib"
PCA_MODEL_FILE = "pca_model.joblib"

# --- These are our 4 "expert" features ---
CUSTOM_FEATURES = [
    'total_word_count',
    'turn_count',
    'salesman_talk_ratio',
    'customer_question_count'
]
N_COMPONENTS = 30 # Must match what we trained on
# ---------------------

print("--- Phase 2 (Final): Saving the Preprocessing Pipeline ---")

# 1. Load original split data
print(f"Loading '{X_TRAIN_FILE}'...")
if not os.path.exists(X_TRAIN_FILE):
    print(f"\n--- ❌ ERROR ---")
    print(f"File not found: '{X_TRAIN_FILE}'")
    print("Please make sure you have run 'phase_1_split.py' successfully.")
    exit()

X_train_df = pd.read_csv(X_TRAIN_FILE)
print(f"Loaded {X_train_df.shape[0]} training rows.")

# 2. Separate Custom Features from Embedding Features
print("Separating custom features from embedding features...")
embed_features = [col for col in X_train_df.columns if col not in CUSTOM_FEATURES]
X_train_custom = X_train_df[CUSTOM_FEATURES]
X_train_embed = X_train_df[embed_features]

# 3. Fit and SAVE the Custom Feature Scaler
print("\nFitting and saving 'scaler_custom.joblib'...")
scaler_custom = StandardScaler()
scaler_custom.fit(X_train_custom) # Fit on the 4 custom features
joblib.dump(scaler_custom, SCALER_CUSTOM_FILE)

# 4. Fit and SAVE the Embedding Scaler
print("Fitting and saving 'scaler_embed.joblib'...")
scaler_embed = StandardScaler()
scaler_embed.fit(X_train_embed) # Fit on the 384 embedding features
joblib.dump(scaler_embed, SCALER_EMBED_FILE)

# 5. Fit and SAVE the PCA Model
print("Fitting and saving 'pca_model.joblib'...")
# We must scale the embeddings *before* fitting PCA
X_train_embed_scaled = scaler_embed.transform(X_train_embed)

pca = PCA(n_components=N_COMPONENTS, random_state=42)
pca.fit(X_train_embed_scaled) # Fit PCA on the scaled 384 embeddings
joblib.dump(pca, PCA_MODEL_FILE)

print(f"\nTotal explained variance by {N_COMPONENTS} components: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

print("\n---")
print("✅ SUCCESS! ---")
print("Your 3 pipeline files (scalers and PCA) are saved.")
print("We are now 100% ready to build the Streamlit app.")
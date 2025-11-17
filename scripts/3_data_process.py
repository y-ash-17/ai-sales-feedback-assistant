import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Configuration ---
# These are the files we created in Phase 1
X_TRAIN_FILE = "X_train.csv"
X_TEST_FILE = "X_test.csv"
Y_TRAIN_FILE = "y_train.csv"
Y_TEST_FILE = "y_test.csv"

# These are the NEW files we will CREATE
X_TRAIN_OUT_FILE = "X_train_processed_pca.npy"
X_TEST_OUT_FILE = "X_test_processed_pca.npy"

# --- These are our 4 "expert" features ---
CUSTOM_FEATURES = [
    'total_word_count',
    'turn_count',
    'salesman_talk_ratio',
    'customer_question_count'
]
# ---------------------

print("--- Phase 2 (New): Preprocessing with PCA ---")

# 1. Load original split data
print(f"Loading '{X_TRAIN_FILE}' and '{X_TEST_FILE}'...")
if not os.path.exists(X_TRAIN_FILE) or not os.path.exists(X_TEST_FILE):
    print(f"\n--- ❌ ERROR ---")
    print(f"File not found: '{X_TRAIN_FILE}' or '{X_TEST_FILE}'")
    print("Please make sure you have run 'phase_1_split.py' first.")
    exit()

X_train_df = pd.read_csv(X_TRAIN_FILE)
X_test_df = pd.read_csv(X_TEST_FILE)
print(f"Loaded {X_train_df.shape[0]} training rows and {X_test_df.shape[0]} testing rows.")

# 2. Separate Custom Features from Embedding Features
print("Separating custom features from embedding features...")
# Get embedding feature names (e.g., 'embed_0', 'embed_1', ...)
embed_features = [col for col in X_train_df.columns if col not in CUSTOM_FEATURES]

X_train_custom = X_train_df[CUSTOM_FEATURES]
X_train_embed = X_train_df[embed_features]

X_test_custom = X_test_df[CUSTOM_FEATURES]
X_test_embed = X_test_df[embed_features]

print(f"Custom features shape: {X_train_custom.shape}")
print(f"Embedding features shape: {X_train_embed.shape}")

# 3. Apply StandardScaler (We need two scalers, one for each set)
print("\nApplying StandardScaler to custom features...")
scaler_custom = StandardScaler()
X_train_custom_scaled = scaler_custom.fit_transform(X_train_custom)
X_test_custom_scaled = scaler_custom.transform(X_test_custom)

print("Applying StandardScaler to embedding features...")
scaler_embed = StandardScaler()
X_train_embed_scaled = scaler_embed.fit_transform(X_train_embed)
X_test_embed_scaled = scaler_embed.transform(X_test_embed)

# 4. Apply PCA (Your great idea!)
# We will compress the 384 embedding features down to 30
N_COMPONENTS = 30
print(f"\nApplying PCA to embedding features (compressing 384 -> {N_COMPONENTS})...")

pca = PCA(n_components=N_COMPONENTS, random_state=42)
# We FIT *only* on the training data
X_train_pca = pca.fit_transform(X_train_embed_scaled)
# We TRANSFORM the test data using the *same* fit
X_test_pca = pca.transform(X_test_embed_scaled)

print(f"New PCA-features shape: {X_train_pca.shape}")
print(f"Total explained variance by {N_COMPONENTS} components: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

# 5. Combine our 4 custom features + 30 PCA features
print("Combining custom features and PCA features...")
# np.hstack stacks arrays horizontally
X_train_final = np.hstack((X_train_custom_scaled, X_train_pca))
X_test_final = np.hstack((X_test_custom_scaled, X_test_pca))

new_feature_count = X_train_final.shape[1]
print(f"Created new feature matrix with {new_feature_count} total features (4 custom + {N_COMPONENTS} PCA).")

# 6. Save the new processed files
print(f"\nSaving new processed files to '{X_TRAIN_OUT_FILE}' and '{X_TEST_OUT_FILE}'...")
np.save(X_TRAIN_OUT_FILE, X_train_final)
np.save(X_TEST_OUT_FILE, X_test_final)

# We are NOT saving the y_train/y_test files because we are NOT using SMOTE.
# The original .csv files from Phase 1 are still our ground truth.

print("\n---")
print("✅ SUCCESS! ---")
print(f"New feature files created with {new_feature_count} features.")
print("You are now ready for the new Phase 3 training script.")
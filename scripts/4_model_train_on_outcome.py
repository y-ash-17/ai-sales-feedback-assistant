import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# --- Configuration ---
# These are the NEW files we created in our "data-centric" Phase 2
X_TRAIN_FILE = "X_train_processed_pca.npy"

# This is the ORIGINAL, UNBALANCED label file from Phase 1
# We are NOT using the ..._processed.npy file because we are not using SMOTE
Y_TRAIN_FILE = "y_train.csv"
TARGET_COLUMN = "outcome"

# This is the file we will CREATE
MODEL_OUTPUT_FILE = "sales_outcome_model_pca.joblib"
# ---------------------

print("--- Phase 3a: Model Training (PCA + Class Weight) ---")

# 1. Load our new PCA-processed training data
print(f"Loading data from '{X_TRAIN_FILE}' and '{Y_TRAIN_FILE}'...")
if not os.path.exists(X_TRAIN_FILE) or not os.path.exists(Y_TRAIN_FILE):
    print(f"\n--- ❌ ERROR ---")
    print(f"File not found: '{X_TRAIN_FILE}' or '{Y_TRAIN_FILE}'")
    print("Please make sure you have run 'phase_2_preprocess_pca.py' first.")
    exit()

X_train = np.load(X_TRAIN_FILE)
y_train_df = pd.read_csv(Y_TRAIN_FILE)
y_train = y_train_df[TARGET_COLUMN]

print(f"Loaded {X_train.shape[0]} training rows with {X_train.shape[1]} features.")
print("Training label distribution (imbalanced):")
print(y_train.value_counts())

# 2. Initialize and Train the Model
print("\nInitializing RandomForestClassifier...")
# --- THIS IS THE NEW IMBALANCE FIX ---
# We are telling the model to "pay more attention" to the
# minority class ('failure') automatically.
model = RandomForestClassifier(
    random_state=42, 
    n_estimators=100, 
    n_jobs=-1,
    class_weight='balanced' # <-- This is our new strategy
)

print("Training the model on the PCA features with class_weight='balanced'...")
# We fit on the imbalanced data. The model handles the bias itself.
model.fit(X_train, y_train)
print("Model training complete.")

# 3. Save the Trained Model to a File
print(f"\nSaving trained model to '{MODEL_OUTPUT_FILE}'...")
joblib.dump(model, MODEL_OUTPUT_FILE)

print("\n---")
print("✅ SUCCESS! ---")
print(f"Your PCA-based model has been trained and saved as '{MODEL_OUTPUT_FILE}'.")
print("\nYou are now ready for the (new) evaluation script.")
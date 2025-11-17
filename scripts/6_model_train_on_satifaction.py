import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier # We are using a Classifier

# --- Configuration ---
# These are the files we load
X_TRAIN_FILE = "X_train_processed_pca.npy"
Y_TRAIN_FILE = "y_train.csv" # The original, unbalanced label file
TARGET_COLUMN = "satisfaction" 

# This is the NEW model file we will CREATE
MODEL_OUTPUT_FILE = "sales_satisfaction_model_clf_v2.joblib"
# ---------------------

print("--- Phase 3a: Model Training (Satisfaction Classifier V2 - Binary) ---")

# 1. Load our PCA-processed training data
print(f"Loading data from '{X_TRAIN_FILE}' and '{Y_TRAIN_FILE}'...")
if not os.path.exists(X_TRAIN_FILE) or not os.path.exists(Y_TRAIN_FILE):
    print(f"\n--- ❌ ERROR ---")
    print(f"File not found: '{X_TRAIN_FILE}' or '{Y_TRAIN_FILE}'")
    print("Please make sure you have run 'phase_2_preprocess_pca.py' first.")
    exit()

X_train = np.load(X_TRAIN_FILE)
y_train_df = pd.read_csv(Y_TRAIN_FILE)
y_train_scores = y_train_df[TARGET_COLUMN] # This is a Series of numbers (1-5)

print(f"Loaded {X_train.shape[0]} training rows with {X_train.shape[1]} features.")

# 2. --- THIS IS THE NEW 2-BIN LOGIC ---
# We will convert the 1-5 scores into 2 categories
def bin_satisfaction_scores_v2(score):
    if score <= 3:
        return "Non-Positive" # Merges 1, 2, and 3
    else: # 4 or 5
        return "Positive"

print(f"\nConverting numeric scores (1-5) into 2 categories...")
y_train_binned = y_train_scores.apply(bin_satisfaction_scores_v2)

print("New binned training label distribution (imbalanced):")
print(y_train_binned.value_counts())

# 3. Initialize and Train the Model
print("\nInitializing RandomForestClassifier...")
# We use class_weight='balanced' to handle the new ~3:1 imbalance
model = RandomForestClassifier(
    random_state=42, 
    n_estimators=100, 
    n_jobs=-1,
    class_weight='balanced' # <-- This is our imbalance fix
)

print(f"Training the model to predict new binned categories...")
# We fit on the new "Positive" / "Non-Positive" labels
model.fit(X_train, y_train_binned)
print("Model training complete.")

# 4. Save the Trained Model to a File
print(f"\nSaving trained model to '{MODEL_OUTPUT_FILE}'...")
joblib.dump(model, MODEL_OUTPUT_FILE)

print("\n---")
print("✅ SUCCESS! ---")
print(f"Your *v2 satisfaction classifier* model has been trained and saved as '{MODEL_OUTPUT_FILE}'.")
print("\nYou are now ready for the (new) evaluation script.")
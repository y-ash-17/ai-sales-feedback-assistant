import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --- Configuration ---
# These are the NEW files we will LOAD
MODEL_FILE = "sales_outcome_model_pca.joblib"
X_TEST_FILE = "X_test_processed_pca.npy"
Y_TEST_FILE = "y_test.csv" # The original, human-readable test labels

TARGET_COLUMN = "outcome"
N_PCA_COMPONENTS = 30 # Must match the N_COMPONENTS from Phase 2
CUSTOM_FEATURES = [
    'total_word_count',
    'turn_count',
    'salesman_talk_ratio',
    'customer_question_count'
]
# ---------------------

print("--- Phase 3b: Model Evaluation (PCA + Class Weight Model) ---")

# 1. Check if all required files exist
print("Loading model, encoder, and test data...")
required_files = [MODEL_FILE, X_TEST_FILE, Y_TEST_FILE]
for f in required_files:
    if not os.path.exists(f):
        print(f"\n--- ❌ ERROR ---")
        print(f"File not found: '{f}'")
        print("Please make sure you have run all previous phases successfully.")
        exit()

# 2. Load the Model and Test Data
model = joblib.load(MODEL_FILE)
X_test = np.load(X_TEST_FILE)

# Load the original, human-readable test labels
y_test_df = pd.read_csv(Y_TEST_FILE)
y_test = y_test_df[TARGET_COLUMN]

print(f"Loaded '{MODEL_FILE}'.")
print(f"Loaded {X_test.shape[0]} test rows to evaluate.")

# 3. Make Predictions on "Unseen" Test Data
print("\nMaking predictions on the 'unseen' test set...")
y_pred = model.predict(X_test)
print("Predictions complete.")

# 4. Evaluate Model Performance
print("\n--- PCA-BASED MODEL PERFORMANCE ---")

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)
print(f"Labels: {model.classes_}")
# Let's make it more readable
tn = cm[0][0] # True Negative
fp = cm[0][1] # False Positive
fn = cm[1][0] # False Negative
tp = cm[1][1] # True Positive
print(f"True '{model.classes_[0]}' (TN):   {tn}")
print(f"False '{model.classes_[1]}' (FP):  {fp}")
print(f"False '{model.classes_[0]}' (FN):  {fn}")
print(f"True '{model.classes_[1]}' (TP):   {tp}")

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 5. BONUS: Feature Importance
print("\n--- TOP 10 MOST IMPORTANT FEATURES (PCA Model) ---")

# We must reconstruct the 34 feature names
pca_feature_names = [f'PCA_{i+1}' for i in range(N_PCA_COMPONENTS)]
all_feature_names = CUSTOM_FEATURES + pca_feature_names

# Get the importances from the trained model
importances = model.feature_importances_

feature_importance_df = pd.DataFrame(
    {"feature": all_feature_names, "importance": importances}
)

# Sort by importance (highest first) and print the top 10
print(feature_importance_df.sort_values(by="importance", ascending=False).head(10))

print("\n---")
print("✅ SUCCESS! ---")
print("Phase 3 is complete. You have evaluated the new PCA-based model.")
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
MODEL_FILE = "sales_satisfaction_model_clf_v2.joblib" # <-- The new 2-class model
X_TEST_FILE = "X_test_processed_pca.npy"
Y_TEST_FILE = "y_test.csv" # The original, human-readable test labels

TARGET_COLUMN = "satisfaction" 
N_PCA_COMPONENTS = 30 # Must match the N_COMPONENTS from Phase 2
CUSTOM_FEATURES = [
    'total_word_count',
    'turn_count',
    'salesman_talk_ratio',
    'customer_question_count'
]
# ---------------------

print("--- Phase 3b: Model Evaluation (Satisfaction Classifier V2) ---")

# 1. Check if all required files exist
print("Loading model and test data...")
required_files = [MODEL_FILE, X_TEST_FILE, Y_TEST_FILE]
for f in required_files:
    if not os.path.exists(f):
        print(f"\n--- ❌ ERROR ---")
        print(f"File not found: '{f}'")
        print("Please make sure you have run 'phase_3a_train_satisfaction_clf_v2.py' successfully.")
        exit()

# 2. Load the Model and Test Data
model = joblib.load(MODEL_FILE)
X_test = np.load(X_TEST_FILE)

# Load the original, human-readable test scores
y_test_df = pd.read_csv(Y_TEST_FILE)
y_test_scores = y_test_df[TARGET_COLUMN] # This is a Series of numbers (1-5)

print(f"Loaded '{MODEL_FILE}'.")
print(f"Loaded {X_test.shape[0]} test rows to evaluate.")

# 3. --- THIS IS THE NEW 2-BIN LOGIC ---
# We must apply the *same* logic to our test labels to create the "ground truth"
def bin_satisfaction_scores_v2(score):
    if score <= 3:
        return "Non-Positive" # Merges 1, 2, and 3
    else: # 4 or 5
        return "Positive"

print(f"\nConverting numeric *test* scores (1-5) into 2 categories...")
y_test_binned = y_test_scores.apply(bin_satisfaction_scores_v2)

print("Test label distribution:")
print(y_test_binned.value_counts())

# 4. Make Predictions on "Unseen" Test Data
print("\nMaking predictions on the 'unseen' test set...")
y_pred = model.predict(X_test) # Model will output "Positive", "Non-Positive"
print("Predictions complete.")

# 5. Evaluate Model Performance
print("\n--- SATISFACTION CLASSIFIER (V2) PERFORMANCE ---")

# Accuracy Score
accuracy = accuracy_score(y_test_binned, y_pred)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("\n--- Confusion Matrix ---")
# We use the model's learned classes to ensure the order is correct
cm = confusion_matrix(y_test_binned, y_pred, labels=model.classes_)
print(cm)
print(f"Labels: {model.classes_}")
# Let's make it more readable (assuming 'Non-Positive' is 0, 'Positive' is 1)
# Find which index is which
neg_idx = 0 if model.classes_[0] == 'Non-Positive' else 1
pos_idx = 1 - neg_idx

tn = cm[neg_idx][neg_idx] # True Negative (Correctly guessed 'Non-Positive')
fp = cm[neg_idx][pos_idx] # False Positive (Guessed 'Positive' but was 'Non-Positive')
fn = cm[pos_idx][neg_idx] # False Negative (Guessed 'Non-Positive' but was 'Positive')
tp = cm[pos_idx][pos_idx] # True Positive (Correctly guessed 'Positive')

print(f"True 'Non-Positive' (TN):   {tn}")
print(f"False 'Positive' (FP):  {fp}")
print(f"False 'Non-Positive' (FN):  {fn}")
print(f"True 'Positive' (TP):   {tp}")

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test_binned, y_pred, labels=model.classes_))

# 6. BONUS: Feature Importance
print("\n--- TOP 10 MOST IMPORTANT FEATURES (Satisfaction Classifier V2) ---")

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
print("Phase 3 is complete. You have evaluated your new *2-class satisfaction* model.")
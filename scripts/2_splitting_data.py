import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
INPUT_FILE = "model_training_data_450.csv"

# --- FIX ---
# We now define ALL columns that are "labels"
LABEL_COLUMNS = ["outcome", "satisfaction"] 
# We will stratify by the most important one
STRATIFY_COLUMN = "outcome"
# -----------

TEST_SPLIT_SIZE = 0.20    # 80% for training, 20% for testing
RANDOM_STATE = 42         # Guarantees we get the same split every time
# ---------------------

print("--- Phase 1: Setup & Splitting ---")

# 1. Check if the input file exists
if not os.path.exists(INPUT_FILE):
    print(f"\n--- ❌ ERROR ---")
    print(f"File not found: '{INPUT_FILE}'")
    print("Please make sure you have run the 'feature_engineering.py' script first,")
    print(f"and that '{INPUT_FILE}' is in the same folder as this script.")
    exit()

print(f"Loading data from '{INPUT_FILE}'...")
df = pd.read_csv(INPUT_FILE)

# 2. Define X (features) and y (label)
print(f"Defining features (X) and targets (y = {LABEL_COLUMNS})...")

# --- FIX ---
# y is now a DataFrame containing BOTH label columns
y = df[LABEL_COLUMNS]
# -----------

# X is everything ELSE. We drop the target and any other non-feature columns.
X = df.drop(columns=[
    'conversation_id', # This is an ID, not a feature
    'outcome',         # This is our target label
    'satisfaction'     # This is the *other* target label
])

print(f"Full X shape: {X.shape}") # (450, 388)
print(f"Full y shape: {y.shape}") # (450, 2) <-- Corrected shape

# 3. Perform the Train-Test Split
print(f"\nSplitting data into {1-TEST_SPLIT_SIZE:.0%} train / {TEST_SPLIT_SIZE:.0%} test...")
print(f"Stratifying by '{STRATIFY_COLUMN}' to preserve bias...")

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,  # y is now the DataFrame with 2 columns
    test_size=TEST_SPLIT_SIZE, 
    random_state=RANDOM_STATE,
    # --- FIX ---
    # We explicitly stratify by the 'outcome' column
    stratify=y[STRATIFY_COLUMN]
    # -----------
)

print("Split complete.")

# 4. Verify the split and stratification
print("\n--- Verification ---")
print(f"X_train shape: {X_train.shape}") # (360, 388)
print(f"y_train shape: {y_train.shape}") # (360, 2) <-- Corrected shape
print(f"X_test shape:  {X_test.shape}")  # (90, 388)
print(f"y_test shape:  {y_test.shape}")  # (90, 2) <-- Corrected shape

print("\nOriginal 'outcome' bias:")
# --- FIX ---
# We now refer to the 'outcome' column within the y DataFrame
print(y['outcome'].value_counts(normalize=True).sort_index())

print("\nTraining set 'outcome' bias (should be identical):")
print(y_train['outcome'].value_counts(normalize=True).sort_index())

print("\nTest set 'outcome' bias (should be identical):")
print(y_test['outcome'].value_counts(normalize=True).sort_index())
# -----------


# 5. Save the 4 new DataFrames to CSV files for the next phase
print("\nSaving 4 new files to disk: X_train.csv, y_train.csv, X_test.csv, y_test.csv")
# These lines are now correct and will save y_train/y_test with 2 columns
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n---")
print("✅ SUCCESS! ---")
print("Phase 1 is complete. Your y_train/y_test files now contain BOTH labels.")
import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
print("Imports complete.")


# --- Step 3: Define Custom Feature Functions (Part A) ---
print("Defining custom feature functions...")

def get_total_word_count(text):
    """Calculates the total word count of the entire transcript."""
    return len(str(text).split())

def get_turn_count(text):
    """Calculates the number of turns based on speaker tags."""
    # Define pattern for any speaker tag, making them case-insensitive
    speaker_tag_pattern = r"(Salesman:|Sales Rep:|Customer:)"
    # Find all occurrences of speaker tags
    turns = re.findall(speaker_tag_pattern, str(text), flags=re.IGNORECASE)
    # Each tag signifies a turn, so the count of tags is the number of turns
    # If no turns are found, return 1 as a default for a single continuous piece of text
    return max(1, len(turns))

def get_salesman_talk_ratio(text):
    """Calculates the percentage of words spoken by the 'Salesman'."""
    salesman_words = 0
    customer_words = 0

    # Define patterns for speakers, making them case-insensitive
    salesman_patterns_re = r"(Salesman:|Sales Rep:)"
    customer_pattern_re = r"(Customer:)"

    # Combine all speaker patterns into a single regex for splitting
    combined_speaker_pattern_re = r"(Salesman:|Sales Rep:|Customer:)"
    # Split the text by any speaker pattern, keeping the delimiters
    segments = re.split(combined_speaker_pattern_re, str(text), flags=re.IGNORECASE)

    current_speaker = None
    for segment in segments:
        if not segment or segment.isspace():
            continue

        if re.match(salesman_patterns_re, segment, flags=re.IGNORECASE):
            current_speaker = "salesman"
        elif re.match(customer_pattern_re, segment, flags=re.IGNORECASE):
            current_speaker = "customer"
        else:
            # This segment is actual speech
            words_in_segment = len(segment.split())
            if current_speaker == "salesman":
                salesman_words += words_in_segment
            elif current_speaker == "customer":
                customer_words += words_in_segment
            # If no current_speaker is set yet, or it's an unrecognized segment, ignore for ratio

    total_words = salesman_words + customer_words
    if total_words == 0:
        return 0.5 # Return 0.5 if no words are found to avoid divide-by-zero, as per original logic

    return salesman_words / total_words

def get_customer_question_count(text):
    """Counts how many times the customer asked a question."""
    count = 0
    # Define patterns for speakers, making them case-insensitive
    customer_pattern_re = r"(Customer:)"

    # Split the text by any speaker pattern, keeping the delimiters
    combined_speaker_pattern_re = r"(Salesman:|Sales Rep:|Customer:)"
    segments = re.split(combined_speaker_pattern_re, str(text), flags=re.IGNORECASE)

    current_speaker = None
    for segment in segments:
        if not segment or segment.isspace():
            continue

        if re.match(customer_pattern_re, segment, flags=re.IGNORECASE):
            current_speaker = "customer"
        elif re.match(r"(Salesman:|Sales Rep:)", segment, flags=re.IGNORECASE):
            current_speaker = "other" # Not customer, so set to something else
        else:
            # This segment is actual speech
            if current_speaker == "customer":
                count += segment.count('?')
    return count

print("Functions defined.")


# --- Step 4: Load Your Labeled Data ---
input_file = "combined_conversations_labeled_final_450.csv"
output_file = "model_training_data_450.csv"

if not os.path.exists(input_file):
    print(f"\n--- ❌ ERROR ---")
    print(f"File not found: '{input_file}'")
    print("Please upload your labeled CSV file to the Colab session first.")
else:
    print(f"\nLoading '{input_file}'...")
    df = pd.read_csv(input_file)

    # Clean up the file: drop the empty 'label' column
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # Rename 'text' to 'full_transcript' for clarity
    if 'text' in df.columns:
        df = df.rename(columns={'text': 'full_transcript'})

    print(f"Loaded {len(df)} conversations.")

    # --- Step 5: Run Feature Engineering Pipeline ---

    # --- Part A: Calculate Custom Features ---
    print("\nCalculating 4 custom features...")
    # We use .apply() to run our functions on every row in the 'full_transcript' column

    # Initialize a new DataFrame to hold our features
    df_features_custom = pd.DataFrame()

    df_features_custom['total_word_count'] = df['full_transcript'].apply(get_total_word_count)
    df_features_custom['turn_count'] = df['full_transcript'].apply(get_turn_count)
    df_features_custom['salesman_talk_ratio'] = df['full_transcript'].apply(get_salesman_talk_ratio)
    df_features_custom['customer_question_count'] = df['full_transcript'].apply(get_customer_question_count)

    print("Custom features calculated.")

    # --- Part B: Generate NLP Embeddings ---
    print("\nLoading Sentence Transformer model (all-MiniLM-L6-v2)...")
    # This will download the model (a few hundred MB) the first time
    model_name = 'all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(model_name)
    print("Model loaded.")

    print(f"Generating embeddings for {len(df)} transcripts...")
    print("(This will take a 1-2 minutes. A progress bar will appear.)")

    # Get a list of all transcript strings
    transcripts_list = df['full_transcript'].tolist()

    # Generate embeddings. This is the slow part.
    # The model is much faster if we give it the whole list at once.
    embeddings = embedding_model.encode(transcripts_list, show_progress_bar=True)

    print(f"Embeddings generated. Shape: {embeddings.shape}") # Should be (450, 384)

    # Convert the embeddings (numpy array) into a DataFrame
    embed_col_names = [f'embed_{i}' for i in range(embeddings.shape[1])]
    df_features_embeddings = pd.DataFrame(embeddings, columns=embed_col_names)
    print("Embeddings converted to DataFrame.")

    # --- Step 6: Combine Everything and Save ---
    print("\nCombining all features and labels...")

    # Get the original labels from our input file
    # We also keep 'conversation_id' for reference
    df_labels = df[['conversation_id', 'outcome', 'satisfaction']]

    # Use pd.concat to join everything side-by-side
    # We must reset_index on the labels to ensure they align perfectly
    df_final_numeric = pd.concat([
        df_labels,
        df_features_custom.reset_index(drop=True),
        df_features_embeddings.reset_index(drop=True)
    ], axis=1) # axis=1 means join horizontally (by columns)

    # --- Step 7: Save the Final File ---
    print(f"Saving final numeric dataset to '{output_file}'...")
    df_final_numeric.to_csv(output_file, index=False, encoding='utf-8')

    print("\n---")
    print("✅ SUCCESS! ---")
    print(f"Your new file '{output_file}' is ready and saved in Colab.")
    print("It contains all 388 features (4 custom + 384 embeddings) and your 2 label columns.")

    print("\n--- Final Dataset Info ---")
    df_final_numeric.info()

    print("\n--- Final Dataset Head (First 5 Rows, First 10 Features) ---")
    # We only show the first 10 feature columns so it fits on the screen
    print(df_final_numeric.head(5).iloc[:, :10])
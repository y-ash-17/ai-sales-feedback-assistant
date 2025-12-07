from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from google import genai
from sentence_transformers import SentenceTransformer

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- 1. LOAD ALL MODELS & PIPELINE FILES ---
# (This section is unchanged)
@st.cache_resource
def load_models_and_pipeline():
    """
    Loads all 5 pipeline/model files from disk into memory.
    """
    print("--- Loading all models and pipeline objects... ---")
    
    required_files = [
        "scaler_custom.joblib", "scaler_embed.joblib", "pca_model.joblib",
        "sales_outcome_model_pca.joblib", "sales_satisfaction_model_clf_v2.joblib"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Error: Missing required model files: {', '.join(missing_files)}. Please run 'phase_2_save_pipeline.py' first.")
        st.stop()
    
    pipeline = {
        "scaler_custom": joblib.load("scaler_custom.joblib"),
        "scaler_embed": joblib.load("scaler_embed.joblib"),
        "pca_model": joblib.load("pca_model.joblib")
    }
    models = {
        "outcome_model": joblib.load("sales_outcome_model_pca.joblib"),
        "satisfaction_model": joblib.load("sales_satisfaction_model_clf_v2.joblib")
    }
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("--- All models loaded successfully. ---")
    return pipeline, models, embedding_model

pipeline, models, embedding_model = load_models_and_pipeline()

# --- 2. DEFINE THE FEATURE ENGINEERING FUNCTIONS ---
# (This section is unchanged)
def get_total_word_count(text):
    return len(str(text).split())

def get_turn_count(text):
    return text.count('\n') + 1

def get_salesman_talk_ratio(text):
    salesman_words = 0
    customer_words = 0
    total_words = 0
    salesman_patterns = ["Salesman:", "Sales Rep:"]
    customer_patterns = ["Customer:"]
    lines = str(text).split('\n')
    for line in lines:
        if any(line.strip().startswith(p) for p in salesman_patterns):
            salesman_words += len(line.split())
        elif any(line.strip().startswith(p) for p in customer_patterns):
            customer_words += len(line.split())
    total_words = salesman_words + customer_words
    if total_words == 0:
        return 0.5
    return salesman_words / total_words

def get_customer_question_count(text):
    count = 0
    lines = str(text).split('\n')
    for line in lines:
        if line.strip().startswith("Customer:"):
            count += line.count('?')
    return count

def process_new_transcript(raw_text, pipeline, embedding_model):
    print("Processing new transcript...")
    custom_features_dict = {
        'total_word_count': get_total_word_count(raw_text),
        'turn_count': get_turn_count(raw_text),
        'salesman_talk_ratio': get_salesman_talk_ratio(raw_text),
        'customer_question_count': get_customer_question_count(raw_text)
    }
    custom_features_array = np.array([[
        custom_features_dict['total_word_count'],
        custom_features_dict['turn_count'],
        custom_features_dict['salesman_talk_ratio'],
        custom_features_dict['customer_question_count']
    ]])
    embedding_vector = embedding_model.encode([raw_text])
    custom_scaled = pipeline["scaler_custom"].transform(custom_features_array)
    embed_scaled = pipeline["scaler_embed"].transform(embedding_vector)
    embed_pca = pipeline["pca_model"].transform(embed_scaled)
    final_feature_vector = np.hstack((custom_scaled, embed_pca))
    print(f"Final feature vector created. Shape: {final_feature_vector.shape}")
    return final_feature_vector, custom_features_dict


# # --- 3. NEW LLM-BASED FEEDBACK FUNCTION ---
# def generate_llm_feedback(outputs, features):
#     """
#     Connects to Ollama and generates dynamic, detailed feedback.
#     """
    
#     # 1. Build the "Smarter" Prompt from our test script
#     system_prompt = (
#         "You are 'Coach AI', an expert sales call reviewer. Your feedback is "
#         "professional, encouraging, and highly specific."
#         "You must follow this exact format:\n"
#         "1. Start with a 1-sentence 'Overall Call Summary:' (e.g., 'Good Call', 'Missed Opportunity', 'Tough Call').\n"
#         "2. Create a markdown-formatted section titled '✅ What went well:'. "
#         "   - Provide 1-2 bullet points. If nothing went well, say so professionally.\n"
#         "3. Create a markdown-formatted section titled '💡 Where you can improve:'. "
#         "   - Provide 2-3 actionable bullet points.\n"
#         "4. **Crucially:** You MUST use the specific data from the 'Key Call Metrics' to "
#         "   justify your points. For example, if 'Salesman Talk Ratio' is high, explain *why* "
#         "   that's bad and connect it to the 'Predicted Outcome'.\n"
#         "5. The total feedback should be detailed, around 150-200 words."
#     )
    
#     user_prompt = f"""
#     Here is the analysis of my sales call. Please provide detailed, actionable feedback.

#     **1. AI Model Predictions:**
#     - Predicted Call Outcome: {outputs['outcome']}
#     - Predicted Customer Satisfaction: {outputs['satisfaction']}

#     **2. Key Call Metrics:**
#     - Salesman Talk Ratio: {features['salesman_talk_ratio'] * 100:.0f}%
#     - Total Word Count: {features['total_word_count']}
#     - Total Call Turns (Back-and-Forth): {features['turn_count']}
#     - Customer Questions Asked: {features['customer_question_count']}
#     """
    
#     # 2. Call the Ollama API (with error handling)
#     try:
#         response = ollama.chat(
#             model='mistral',
#             messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': user_prompt},
#             ]
#         )
#         return response['message']['content']
    
#     except Exception as e:
#         # This is a critical error message for the user
#         st.error(f"❌ **Ollama Connection Error:** {e}")
#         st.error("Please make sure your local Ollama server is running. In a separate terminal, run `ollama run mistral` and try again.")
#         return None

def generate_llm_feedback(outputs, features):
    """
    Uses Google Gemini to create structured and detailed sales coaching feedback.
    """
    prompt = f"""
You are 'Coach AI', an expert sales performance coach. Provide helpful, clear and structured advice.

Follow this exact structure:
1. **Overall Call Summary** (1 sentence)
2. **What went well** (2–3 bullet points)
3. **Where to improve** (3 bullet points, actionable)
4. **Scores**
   - Professionalism (0–10)
   - Conversion likelihood (0–10)

Here is the call analysis:

Predicted Outcome: {outputs['outcome']}
Predicted Customer Satisfaction: {outputs['satisfaction']}

Key Call Metrics:
- Salesman Talk Ratio: {features['salesman_talk_ratio'] * 100:.0f}%
- Total Word Count: {features['total_word_count']}
- Total Call Turns: {features['turn_count']}
- Customer Questions: {features['customer_question_count']}

Use the data above when giving feedback.
"""

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return None


# --- 4. BUILD THE STREAMLIT UI ---

st.set_page_config(page_title="Sales Call Analyzer", layout="wide")
st.title("🤖 AI Sales Call Coach")
st.write("This tool uses two ML models to analyze your call, then feeds those results to a LLM to generate actionable feedback.")

# Create a sample transcript
SAMPLE_TRANSCRIPT = """Customer: Hi, Im interested in learning more about your health products.
Salesman: Great! Im happy to help. Tell me, what specific health concerns do you have?
Customer: Ive been experiencing digestive issues lately and Im looking for a solution.
Salesman: I understand how frustrating that can be. Many of our customers have found relief with our digestive health supplements. Would you like me to provide more information?
Customer: Ive tried different products before, but nothing seems to work. Im skeptical.
Salesman: I completely understand your skepticism. Its important to find the right solution that works for you. Our digestive health supplements are backed by scientific research and have helped many people with similar issues. Would you be open to trying them?
Customer: Im concerned about the potential side effects of the supplements. Are they safe?
Salesman: Safety is our top priority. Our digestive health supplements are made with natural ingredients and undergo rigorous testing to ensure their safety and effectiveness. We can provide you with detailed information on the ingredients and any potential side effects. Would that help alleviate your concerns?
Customer: Im still unsure. Can you share some success stories from your customers?
Salesman: Absolutely! We have numerous success stories from customers who have experienced significant improvements in their digestive health after using our supplements. I can provide you with testimonials and reviews to give you a better idea of the positive results people have achieved. Would you like to hear some of their stories?
Customer: I appreciate your assistance. Ill take some time to think about it before making a decision.
Salesman: Of course, take all the time you need. Remember, building rapport is important to us, so feel free to reach out if you have any more questions or if theres anything else I can help you with."""

# Get user input
raw_text_input = st.text_area(
    "Paste a full raw transcript here (must include 'Customer:' and 'Salesman:' or 'Sales Rep:')",
    height=300,
    value=SAMPLE_TRANSCRIPT # Pre-fill with our sample
)

analyze_button = st.button("Analyze Call Transcript")

if analyze_button:
    if len(raw_text_input) < 50:
        st.error("Error: Transcript is too short. Please paste a full conversation.")
    else:
        # Show a spinner while processing
        with st.spinner("Analyzing... (Running ML models)"):
            
            # 1. Run the full processing pipeline
            final_feature_vector, custom_features = process_new_transcript(
                raw_text_input, pipeline, embedding_model
            )
            
            # 2. Get predictions from BOTH models
            outcome_pred = models["outcome_model"].predict(final_feature_vector)[0]
            satisfaction_pred = models["satisfaction_model"].predict(final_feature_vector)[0]
            
            # --- 3. Generate LLM Feedback ---
            st.subheader("Actionable Feedback (from Coach AI)")
            model_outputs = {
                "outcome": outcome_pred,
                "satisfaction": satisfaction_pred
            }
            # This now calls our new LLM function
            feedback_text = generate_llm_feedback(model_outputs, custom_features)
            
            if feedback_text:
                st.markdown(feedback_text)
            
            st.divider()

            # --- 4. Display the raw results ---
            st.subheader("Raw Model & Feature Data")
            
            # Use columns for a clean layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if outcome_pred == 'success':
                    st.metric("Predicted Outcome", "✅ Success")
                else:
                    st.metric("Predicted Outcome", "❌ Failure")

            with col2:
                if satisfaction_pred == 'Positive':
                    st.metric("Predicted Satisfaction", "😊 Positive")
                else:
                    st.metric("Predicted Satisfaction", "🙁 Non-Positive" )
            
            with col3:
                st.metric(
                    "Salesman Talk Ratio", 
                    f"{custom_features['salesman_talk_ratio'] * 100:.0f}%",
                    help="The % of words spoken by the salesman. This was the #1 most important feature."
                )
            
            st.divider()
            
            st.subheader("Calculated Features")
            # Show the raw features that were calculated
            st.json(custom_features)
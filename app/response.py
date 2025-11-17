import ollama
import json

# --- 1. Define Mock Data ---
# This is what our Streamlit app will send to the LLM.
# We're using the same "bad call" example as before.
mock_model_outputs = {
    "outcome": "failure",
    "satisfaction": "Non-Positive"
}

mock_key_features = {
    'total_word_count': 280,
    'turn_count': 8,
    'salesman_talk_ratio': 0.82,  # Very high (bad)
    'customer_question_count': 1
}

# --- 2. Build the NEW, "Smarter" LLM Prompt ---
def build_prompt_v2(outputs, features):
    """
    Creates a detailed system prompt (persona) and user prompt (data)
    inspired by the user's screenshot.
    """
    
    # The System Prompt is now much more specific and demanding.
    system_prompt = (
        "You are 'Coach AI', anert sales call reviewer. Your feedback is "
        "professional, encouraging, and highly specific."
        "You must follow this exact format:\n"
        "1. Start with a 1-sentence 'Overall Call Summary:' (e.g., 'Good Call', 'Missed Opportunity', 'Tough Call').\n"
        "2. Create a markdown-formatted section titled '✅ What went well:'. "
        "   - Provide 1-2 bullet points. If nothing went well, say so professionally.\n"
        "3. Create a markdown-formatted section titled '💡 Where you can improve:'. "
        "   - Provide 2-3 actionable bullet points.\n"
        "4. **Crucially:** You MUST use the specific data from the 'Key Call Metrics' to "
        "   justify your points. For example, if 'Salesman Talk Ratio' is high, explain *why* "
        "   that's bad and connect it to the 'Predicted Outcome'.\n"
        "5. The total feedback should be detailed, around 150-200 words."
    )
    
    # The User Prompt now includes ALL 4 of our custom features.
    user_prompt = f"""
    Here is the analysis of my sales call. Please provide detailed, actionable feedback.

    **1. AI Model Predictions:**
    - Predicted Call Outcome: {outputs['outcome']}
    - Predicted Customer Satisfaction: {outputs['satisfaction']}

    **2. Key Call Metrics:**
    - Salesman Talk Ratio: {features['salesman_talk_ratio'] * 100:.0f}%
    - Total Word Count: {features['total_word_count']}
    - Total Call Turns (Back-and-Forth): {features['turn_count']}
    - Customer Questions Asked: {features['customer_question_count']}
    """
    
    return system_prompt, user_prompt

system_prompt, user_prompt = build_prompt_v2(mock_model_outputs, mock_key_features)

# --- 3. Connect to Ollama and Get Response ---
print(f"Connecting to Ollama and sending *new* prompt to 'mistral' model...")
print("---")
print(f"SYSTEM PROMPT: {system_prompt}")
print(f"\nUSER PROMPT: {user_prompt}")
print("---")

try:
    # This sends the prompt to your locally running Ollama server
    response = ollama.chat(
        model='mistral',  # Make sure you have 'mistral' pulled in Ollama
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
    )
    
    print("\n✅ Success! Received response from Ollama (V2):")
    print("---")
    # The response content is nested in this dictionary
    print(response['message']['content'])
    print("---")

except Exception as e:
    print(f"\n❌ ERROR: Could not connect to Ollama.")
    print("Please make sure Ollama is running on your local machine.")
    print("In a separate terminal, run 'ollama run mistral' and then try this script again.")
    print(f"\nFull error details: {e}")
import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- 1. SETTINGS ---
MAX_ENC_LEN = 14  

# --- 2. LOAD TOKENIZER ---
try:
    with open('tokenizer.pkl', 'rb') as obj:
        tokenizer = pickle.load(obj)
except Exception as e:
    st.error(f"Tokenizer not found: {e}")
    st.stop()

# --- 3. LOAD INFERENCE MODELS ---
# We load the specific encoder and decoder models saved from Colab
try:
    # Use compile=False to avoid any optimizer/argument mismatch errors
    encoder_model = load_model('encoder_model.keras', compile=False)
    decoder_model = load_model('decoder_model.keras', compile=False)
    st.success("Shakespearean brain loaded!")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.info("Ensure encoder_model.keras and decoder_model.keras are in your GitHub repo.")
    st.stop()

# --- 4. INFERENCE LOGIC ---
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=MAX_ENC_LEN, padding='post')

    # Get initial states from encoder
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Ensure states_value is a list [h, c]
    if not isinstance(states_value, list):
        states_value = list(states_value)

    target_seq = np.zeros((1, 1))
    # Standard Seq2Seq models use 'start' or 'bos'
    start_token = tokenizer.word_index.get('<start>') or tokenizer.word_index.get('start')
    
    if start_token:
        target_seq[0, 0] = start_token
    else:
        st.error("Missing start token in tokenizer!")
        return ""

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        # Decoder predicts next word and NEW states
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, None)

        if sampled_word == '<end>' or sampled_word == 'end' or len(decoded_sentence) > 20:
            stop_condition = True
        elif sampled_word is not None:
            decoded_sentence.append(sampled_word)

        # Update target sequence and states for next loop
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return " ".join(decoded_sentence)

# --- 5. STREAMLIT UI ---
st.title("🎭 Shakespearean Next Word Predictor")
st.markdown("---")

input_text = st.text_input("Enter your line:", "To be or not to")

if st.button("Generate Response"):
    if input_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner('Consulting the Bard...'):
            response = generate_response(input_text)
            st.subheader("The Bard says:")
            st.info(response if response else "...silence.")
import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- 1. SETTINGS (IMPORTANT) ---
# Check your training notebook for this value!
MAX_ENC_LEN = 20  # Set this to your actual training length

# --- 2. LOAD TOKENIZER ---
try:
    with open('tokenizer.pkl', 'rb') as obj:
        tokenizer = pickle.load(obj)
except Exception as e:
    st.error(f"Tokenizer not found: {e}")
    st.stop()

# --- 3. LOAD MODEL SAFELY ---
custom_objects = {"NotEqual": tf.math.not_equal}

try:
    # Ensure this file exists on GitHub (unzipped)
    model = load_model('encdec_hamlet.h5', custom_objects=custom_objects)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model load failed: {e}. Check if encdec_hamlet.h5 is in the repo.")
    st.stop() # This prevents the NameError: 'model' is not defined

# --- 4. RE-EXTRACT ENCODER/DECODER ---
# Encoder Surgery
encoder_inputs = model.input[0] 
encoder_lstm = model.get_layer('lstm') 
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm.output
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# Decoder Surgery
latent_dim = 256 
decoder_inputs = model.input[1] 

decoder_state_input_h = Input(shape=(latent_dim,), name="input_h_inf")
decoder_state_input_c = Input(shape=(latent_dim,), name="input_c_inf")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_layer = model.get_layer('embedding_1')
dec_lstm_layer = model.get_layer('lstm_1')
dec_dense_layer = model.get_layer('dense')

dec_emb_inf = dec_emb_layer(decoder_inputs)
decoder_outputs, state_h, state_c = dec_lstm_layer(
    dec_emb_inf, initial_state=decoder_states_inputs
)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [dec_dense_layer(decoder_outputs)] + [state_h, state_c]
)

# --- 5. INFERENCE LOGIC ---
def generate_response(input_text):
    # Prepare input
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=MAX_ENC_LEN, padding='post')

    # Get context from Encoder
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Start decoder with <start> token
    target_seq = np.zeros((1, 1))
    if '<start>' in tokenizer.word_index:
        target_seq[0, 0] = tokenizer.word_index['<start>']
    else:
        st.error("Tokenizer missing '<start>' token!")
        return ""

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, None)

        if sampled_word == '<end>' or len(decoded_sentence) > 20:
            stop_condition = True
        elif sampled_word is not None:
            decoded_sentence.append(sampled_word)

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return " ".join(decoded_sentence)

# --- 6. STREAMLIT UI ---
st.title("Shakespearean Next Word Predictor")
st.write("Enter a sequence of words to generate a response.")

input_text = st.text_input("Enter the sequence", "To be or not to")

if st.button("Predict"):
    with st.spinner('Thinking like Hamlet...'):
        response = generate_response(input_text)
        st.subheader("Generated Response:")
        st.write(response)
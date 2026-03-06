import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
#Load Model

model = load_model('encdec_model.h5') # Since unable to upload model in github due to size

# Load Tokenizer
with open('tokenizer.pkl','rb') as obj:
    tokenizer = pickle.load(obj)

# Function to predict the next word

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# 1. Load the big trained model
model = load_model('encdec_hamlet.h5')

# 2. Re-extract the Encoder
# We grab the input and the internal states (h and c) from the trained layers
encoder_inputs = model.input[0] # encoder_input_data
encoder_outputs, state_h_enc, state_c_enc = model.get_layer('lstm').output 
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# 3. Re-extract the Decoder
# 3. Re-extract Decoder (The 'Surgery' part)
latent_dim = 256 # Same as your training
decoder_inputs = model.input[1] # This is the decoder_input_data placeholder

# Create placeholders for the states we will pass in manually during the loop
decoder_state_input_h = Input(shape=(latent_dim,), name="input_h")
decoder_state_input_c = Input(shape=(latent_dim,), name="input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Grab the layers by their names from the trained model
dec_emb_layer = model.get_layer('embedding_1') 
dec_lstm_layer = model.get_layer('lstm_1')
dec_dense_layer = model.get_layer('dense')

# Re-connect the layers for the inference flow
# Word ID -> Embedding -> LSTM (using manual state inputs) -> Dense
dec_emb_inf = dec_emb_layer(decoder_inputs)
decoder_outputs, state_h, state_c = dec_lstm_layer(
    dec_emb_inf, initial_state=decoder_states_inputs
)

decoder_states = [state_h, state_c]
decoder_outputs = dec_dense_layer(decoder_outputs)

# Define the standalone Decoder Model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

def generate_response(input_text,max_encoder_seq_length=None):
    # 1. Prepare the input: Tokenize and Pad
    # Use the same max_len you used for encoder training
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')

    # 2. Get the "Context Vector" (the thought) from the Encoder
    states_value = encoder_model.predict(input_seq, verbose=0)

    # 3. Start the decoder with the ID of the '<start>' token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        # Predict the next word and get the updated memory states
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Get the ID of the word with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # Convert that ID back into a word string
        sampled_word = tokenizer.index_word.get(sampled_token_index, None)

        # If we hit the '<end>' token or the sentence gets too long, stop
        if sampled_word == '<end>' or len(decoded_sentence) > 20:
            stop_condition = True
        elif sampled_word is not None:
            decoded_sentence.append(sampled_word)

        # Update the target sequence to be the word we JUST predicted
        # This will be fed as the input for the next step
        target_seq[0, 0] = sampled_token_index

        # Update the internal "memory" (states) to pass into the next step
        states_value = [h, c]

    return " ".join(decoded_sentence)


#Streamlit Web APP
st.write("Next word prediction using LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words", "To be or not to")
if st.button("Predict Next Word"):
  #max_seq_len = model.input_shape[1]+1
  next_word = generate_response(input_text)
  st.write(f"Next Word is: {next_word}")
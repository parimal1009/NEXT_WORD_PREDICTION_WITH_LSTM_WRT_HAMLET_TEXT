import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Next Word Prediction With LSTM and Early Stopping (Hamlet Text)")

# Load the LSTM Model
try:
    model = load_model('next_word_lstm.h5')
    st.success("Model loaded successfully!")
except ValueError as ve:
    st.error(f"ValueError while loading model: {ve}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    st.success("Tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list[-(max_sequence_len - 1):]], 
                                    maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Input and Prediction
input_text = st.text_input("Enter a sequence of words:", "To be or not to")
if st.button("Predict Next Word"):
    if model and tokenizer:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.write(f"Next word: **{next_word}**")
        else:
            st.warning("Could not predict the next word. Try a different input.")
    else:
        st.error("Model or tokenizer not loaded properly.")

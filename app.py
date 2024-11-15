import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("MBTI Personality Prediction App")

# Input fields for each feature expected by the model
words_per_comment = st.number_input("Words per comment:", min_value=0.0)
http_per_comment = st.number_input("Links per comment:")
music_per_comment = st.number_input("Mentions of 'music' per comment:")
question_per_comment = st.number_input("Questions per comment:")
img_per_comment = st.number_input("Images per comment:")
excl_per_comment = st.number_input("Exclamation marks per comment:")
ellipsis_per_comment = st.number_input("Ellipses per comment:")

# Prepare data for prediction
input_data = pd.DataFrame([[words_per_comment, http_per_comment, music_per_comment,
                            question_per_comment, img_per_comment, excl_per_comment,
                            ellipsis_per_comment]], 
                          columns=['words_per_comment', 'http_per_comment', 'music_per_comment',
                                   'question_per_comment', 'img_per_comment', 'excl_per_comment',
                                   'ellipsis_per_comment'])

# Button to trigger prediction
if st.button("Predict"):
    # Make the prediction using the loaded model
    prediction = model.predict(input_data)
    st.write("Predicted Personality Type:", prediction[0])

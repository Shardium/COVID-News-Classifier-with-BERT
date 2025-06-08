import streamlit as st
import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import os

# Rebuild the model architecture
model = TFAutoModel.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]
        return self.fc(x)

classifier = BERTForClassification(model)

# Dummy input to build the model's weights
dummy_inputs = {
    "input_ids": tf.zeros((1, 100), dtype=tf.int32),
    "attention_mask": tf.zeros((1, 100), dtype=tf.int32)
}
classifier(dummy_inputs)

weights_path = "model_weights.h5"
if os.path.exists(weights_path):
    classifier.load_weights(weights_path)
    st.success("File is found.")
else:
    st.warning("Warning: Weights file not found. Model will produce random outputs.")


MAX_LENGTH = 100


#---------------------------------------------------------------------------------------------------


# UI
# st.title("Fake News Detector (COVID-19 Edition)")
# user_input = st.text_area("Enter a tweet or news snippet:")

# if st.button("Classify"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         inputs = tokenizer(
#             user_input,
#             return_tensors="tf",
#             padding='max_length',
#             truncation=True,
#             max_length=MAX_LENGTH
#         )

#         prediction = classifier.predict({
#             "input_ids": inputs["input_ids"],
#             "attention_mask": inputs["attention_mask"],
#         })

#         confidence = prediction.item()
#         label = "Real" if confidence > 0.5 else "Fake"
#         st.success(f"Prediction: **{label}** ({confidence:.2%} confidence) | raw confidence: {confidence}")

#----------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="ADE News Detections",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    body {
        background-color: #2e2e2e;
        color: white;
    }
    .stApp {
        background-color: #2e2e2e;
        color: white;
    }
    textarea {
        background-color: #444444 !important;
        color: white !important;
    }
    .stTextArea label {
        font-size: 20px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Fake News Detector (COVID-19 Edition)")
user_input = st.text_area("Enter a tweet or news snippet:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="tf",
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )

        prediction = classifier.predict({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })

        confidence = prediction.item()
        label = "Real" if confidence > 0.5 else "Fake"
        st.success(f"Prediction: **{label}** ({confidence:.2%} confidence) | raw confidence: {confidence}")


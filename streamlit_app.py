import streamlit as st
import torch
from config import get_config
from inference import load_model, translate_sentence

st.title("Machine Translation with Transformer")

config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, tokenizer_src, tokenizer_tgt = load_model(config, device)

st.write("Enter a sentence to translate:")

input_sentence = st.text_area("Source Sentence", "Type here...")
if st.button("Translate"):
    with st.spinner("Translating..."):
        translation = translate_sentence(model, input_sentence, tokenizer_src, tokenizer_tgt, config, device)
        st.write("Translation:")
        st.write(translation)

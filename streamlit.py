import streamlit as st
import torch
import torch.nn.functional as F
from model import GPTLanguageModel, Block, MultiHeadAttention, FeedFoward, Head  # Import all necessary classes


# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPTLanguageModel()
model.load_state_dict(torch.load('model_state_dict.pt',map_location=device))
model.eval()

#read to inspect
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

#unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = {ch:i for i,ch in enumerate(chars)}
itos ={i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] ##encoder
decode = lambda l: ''.join([itos[i] for i in l]) ##decoder




# Streamlit app
st.title('Shakespeare GPT')

# Number of words input
num_words = st.number_input('Enter the number of words to generate:', min_value=1, step=1)

# Generate text button
if st.button('Generate Text'):
    if num_words > 0:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_text = model.generate(context, max_new_tokens=num_words)
        decoded_text = decode(generated_text[0].tolist())
        st.text_area('Generated Text', value=decoded_text, height=300)
    else:
        st.warning('Please enter a valid number of words.')
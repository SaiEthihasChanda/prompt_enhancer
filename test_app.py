import streamlit as st
import json
import torch
import torch.nn as nn
import random
from nltk.tokenize import word_tokenize
import nltk
import synonyms

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Define the Seq2Seq Model (must match training script)
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True, dropout=0.3)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.fc.out_features
        
        # Encoder
        embedded = self.embedding(src)
        enc_output, (hidden, cell) = self.encoder(embedded)
        
        # Decoder
        outputs = torch.zeros(batch_size, tgt_len-1, vocab_size).to(src.device)
        dec_input = tgt[:, 0].unsqueeze(1)  # Start with <SOS>
        dec_hidden = (hidden, cell)
        
        for t in range(tgt_len-1):
            dec_embed = self.embedding(dec_input)
            dec_output, dec_hidden = self.decoder(dec_embed, dec_hidden)
            output = self.fc(dec_output.squeeze(1))
            outputs[:, t, :] = output
            
            # Teacher forcing (not used in inference)
            if random.random() < teacher_forcing_ratio:
                dec_input = tgt[:, t+1].unsqueeze(1)
            else:
                dec_input = output.argmax(1).unsqueeze(1)
        
        return outputs

# Load Vocabulary
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(item['original'], item['enhanced']) for item in data]

def build_vocab(texts):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    word_count = {}
    for text in texts:
        for word in word_tokenize(text.lower()):
            word_count[word] = word_count.get(word, 0) + 1
    for word, count in word_count.items():
        if count >= 1:  # Include all words
            vocab[word] = len(vocab)
    return vocab, {v: k for k, v in vocab.items()}

def tokenize_and_convert(text, vocab, max_len=50):
    tokens = word_tokenize(text.lower())[:max_len-1]
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    token_ids = [vocab['<SOS>']] + token_ids + [vocab['<EOS>']]
    if len(token_ids) < max_len:
        token_ids += [vocab['<PAD>']] * (max_len - len(token_ids))
    return token_ids

# Inference Function
def enhance_prompt(model, prompt, vocab, inv_vocab, max_len=50, device='cpu'):
    model.eval()
    model = model.to(device)
    token_ids = tokenize_and_convert(prompt, vocab, max_len)
    src = torch.tensor([token_ids]).to(device)
    
    # Encoder
    embedded = model.embedding(src)
    _, (hidden, cell) = model.encoder(embedded)
    
    # Decoder
    dec_input = torch.tensor([[vocab['<SOS>']]]).to(device)
    output_tokens = []
    for _ in range(max_len):
        dec_embed = model.embedding(dec_input)
        dec_output, (hidden, cell) = model.decoder(dec_embed, (hidden, cell))
        output = model.fc(dec_output.squeeze(1))
        pred_token = output.argmax(1).item()
        if pred_token == vocab['<EOS>']:
            break
        output_tokens.append(pred_token)
        dec_input = torch.tensor([[pred_token]]).to(device)
    
    return ' '.join(inv_vocab.get(t, '<UNK>') for t in output_tokens)

# Streamlit App
def main():
    # Set up Streamlit page
    st.title("Prompt Enhancer")
    st.write("Enter a verbose programming-related prompt, and get a concise, standardized version.")
    
    # Load dataset and vocabulary
    dataset_path = 'augmented_programming_prompts.json'
    try:
        data = load_dataset(dataset_path)
    except FileNotFoundError:
        st.error("Error: 'large_programming_prompts.json' not found. Please ensure the file is in the working directory.")
        return
    
    all_texts = [pair[0] for pair in data] + [pair[1] for pair in data]
    vocab, inv_vocab = build_vocab(all_texts)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(vocab_size=len(vocab), embed_size=128, hidden_size=256)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('prompt_enhancer.pt', map_location=device))
    except FileNotFoundError:
        st.error("Error: 'prompt_enhancer.pt' not found. Please train the model first using prompt_enhancer.py.")
        return
    
    # User input
    user_prompt = st.text_area("Enter your prompt:", 
                              placeholder="e.g., Can you please provide a detailed explanation of how to implement a binary search algorithm in Python?",
                              height=100)
    
    if st.button("Enhance Prompt"):
        if user_prompt.strip():
            enhanced_prompt = enhance_prompt(model, user_prompt, vocab, inv_vocab, device=device)
            st.subheader("Enhanced Prompt")
            st.write(enhanced_prompt)
        else:
            st.warning("Please enter a valid prompt.")
    
    # Example prompts
    st.subheader("Try Example Prompts")
    example_prompts = [
        "Can you please provide a detailed explanation of how to implement a binary search algorithm in Python?",
        "I need a step-by-step guide on writing a function to reverse a linked list in C++",
        "How do I create a simple GUI application using Tkinter in Python?"
    ]
    for prompt in example_prompts:
        if st.button(f"Try: {prompt[:50]}..."):
            enhanced = enhance_prompt(model, prompt, vocab, inv_vocab, device=device)
            st.write(f"**Original**: {prompt}")
            st.write(f"**Enhanced**: {enhanced}")

if __name__ == '__main__':
    main()
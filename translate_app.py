import streamlit as st
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.data import Field
import spacy
import numpy as np

# Load spacy tokenizers for German and English
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Define tokenizers
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# Load vocabulary
SRC.build_vocab([['<sos>', '<eos>']])
TRG.build_vocab([['<sos>', '<eos>']])

# Load the model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, 
                 dropout, max_length = 100):
        super().__init__()

        self.src_tok_emb = nn.Embedding(input_dim, hid_dim)
        self.trg_tok_emb = nn.Embedding(output_dim, hid_dim)
        self.positional_encoding = nn.Embedding(max_length, hid_dim)
        
        self.transformer = nn.Transformer(hid_dim, n_heads, n_layers, n_layers, 
                                          pf_dim, dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, M = trg.shape
        
        src_pos = (torch.arange(0, src_seq_len).unsqueeze(1).repeat(1, N)).to(device)
        trg_pos = (torch.arange(0, trg_seq_len).unsqueeze(1).repeat(1, M)).to(device)
        
        src = self.dropout((self.src_tok_emb(src) * self.scale) + self.positional_encoding(src_pos))
        trg = self.dropout((self.trg_tok_emb(trg) * self.scale) + self.positional_encoding(trg_pos))
        
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        
        output = self.transformer(src, trg)
        
        output = self.fc_out(output)
        
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

model = Transformer(INPUT_DIM, OUTPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT).to(device)
model.load_state_dict(torch.load('transformer-model.pt'))
model.eval()

# Function to translate sentences
def translate_sentence(sentence, model, device, max_len=50):
    model.eval()
    tokens = tokenize_de(sentence)
    tokens = [SRC.init_token] + tokens + [SRC.eos_token]
    src_indexes = [SRC.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_mask = model.transformer.generate_square_subsequent_mask(src_tensor.size(0)).to(device)
    with torch.no_grad():
        enc_src = model.transformer.encoder(model.src_tok_emb(src_tensor), src_mask)
    trg_indexes = [TRG.vocab.stoi[TRG.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)
        trg_mask = model.transformer.generate_square_subsequent_mask(trg_tensor.size(0)).to(device)
        with torch.no_grad():
            output = model.transformer.decoder(model.trg_tok_emb(trg_tensor), enc_src, trg_mask)
            pred_token = output.argmax(2)[-1, :].item()
        trg_indexes.append(pred_token)
        if pred_token == TRG.vocab.stoi[TRG.eos_token]:
            break
    trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]
    return ' '.join(trg_tokens[1:-1])

# Streamlit App
st.title('German to English Translation')

st.write("Enter a sentence in German to translate it to English:")

input_sentence = st.text_input("German Sentence:")

if st.button('Translate'):
    translation = translate_sentence(input_sentence, model, device)
    st.write(f'Translation: {translation}')

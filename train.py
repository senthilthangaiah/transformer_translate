import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.metrics import bleu_score
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import random
import math
import time

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load spacy tokenizers for German and English
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_de, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

TRG = Field(tokenize=tokenize_en, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)

# Transformer model definition
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

optimizer = optim.Adam(model.parameters(), lr=0.0005)
PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        
        output = model(src, trg[:-1, :])
        
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:-1, :])
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Loading the best model
model.load_state_dict(torch.load('transformer-model.pt'))

# Evaluation on test data
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

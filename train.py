# Code is based on the youTube Tutorial "Lets build GPT: fron scratch, in code, spelled out." by Andrej Karpathy
# https://www.youtube.com/watch?v=kCc8FmEb1nY&t=429s

#%%

import requests
import os

# Get Tiny Shakespeare Dataset from Git Repository of karpathy and write to folder
response = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

with open('input.txt', 'wb') as f:
    f.write(response.content)

#%%

# Open text file and get length of the file

with open('input.txt', 'r', encoding='utf-8') as f:
    text= f.read()

print("Length of dataset in characters:", len(text))

print(text[:500])

#%%

# Get an idea about the characters used in the text input file

chars = sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

#%%

# Create Lookup table from the character to the integer
stoi = {ch:i for i, ch in enumerate(chars)}

# Create Loopup table from the integer to the character
itos = {i:ch for i, ch in enumerate(chars)}

# Encode Characters and return a list of integers
encode = lambda s: [stoi[c] for c in s]

# Decode list of characters and return a string
decode = lambda l: ''.join(itos[i] for i in l)  

# Test encode and decode functions
print(encode("hi there"))
print(decode(encode('hi there')))

#%%

import torch

data = torch.tensor(encode(text), dtype=torch.long)

# Print Shape and Datatype of data
print(data.shape, data.dtype)

# Print first 1000 encoded characters 
print(data[:1000])

#%%
# Split into train and test data

# Get number of characters in first 90% of text data
n = int(0.9*len(data))

train_data = data[:n]
val_data = data [n:]

#%%
block_size = 8
# block data
x = train_data[:block_size]

# targets
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# %%
# Set parameters for batch processing on the GPU
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will be processed in parallel?
block_size = 8 # maximum context length for predictions?

def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context =xb[b,:t+1]
        target= yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

print (xb)


# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table= nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) #(Batch,Time,Channel)

        if targets == None:
            loss= None
        else:
            B, T, C = logits.shape
            logits= logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets) # Negative Log Likelyhood = Cross Entropy 
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits= logits[:,-1,:] # becomes (B,C)
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next= torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence to the running sequence
            idx= torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx  

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# Create start idx with all zeros, which refers to starting points, with maximum number of characters and decode the result
print( decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens=500)[0].tolist()))

# %%
# create a PyTorch optimizer (Adam is more sophisticated that Gradient Descent), lr: learning rate, 
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) 

# %%
batch_size= 32
for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
  # %%

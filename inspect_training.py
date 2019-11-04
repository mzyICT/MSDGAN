#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('export CUDA_VISIBLE_DEVICES=1')


# In[2]:


import argparse
import os
import math
import random
import json
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from encoder import Encoder
from attention import Attention
from decoder import Decoder
from generator import Generator

from data_loader import SpeechDataset, Padding, ToTensor


# In[3]:


torch.cuda.set_device(1)


# In[4]:


#args = parser.parse_args()


# In[ ]:


with open('labels_dict.json', 'r') as f:
    labels = json.loads(f.read())
    
len(labels)


# In[ ]:


get_ipython().system("ls '../data/'")


# In[ ]:


SIGNAL_SEQ_LEN = 1100 
TXT_SEQ_LEN = 189
OUTPUT_DIM = len(labels)
BATCH_SIZE = 12

audio_conf = {'window': 'hamming',
              'window_size' : 0.02,
              'window_stride' : 0.01,
              'sampling_rate': 16000}

train_dataset = SpeechDataset('train_manifest.csv', 
                            'labels_dict.json',
                            audio_conf,
                            transform=transforms.Compose([Padding(SIGNAL_SEQ_LEN, TXT_SEQ_LEN, 'labels_dict.json')]) 
                              )

val_dataset = SpeechDataset('val_manifest.csv', 
                            'labels_dict.json',
                            audio_conf,
                            transform=transforms.Compose([Padding(SIGNAL_SEQ_LEN, TXT_SEQ_LEN, 'labels_dict.json')]) 
                              )

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)


# In[ ]:


SIGNAL_FEATURE = 161
NUM_GRU = 4
ENC_HID_DIM = 256
DEC_HID_DIM = 256 
DEC_EMB_DIM = 256
DROPOUT_RATE = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

encoder = Encoder(seq_len=SIGNAL_SEQ_LEN, input_size=SIGNAL_FEATURE, 
                  enc_hid_dim=ENC_HID_DIM, num_gru=NUM_GRU, 
                  dec_hid_dim=DEC_HID_DIM, dropout_rate=DROPOUT_RATE, 
                  device=device, use_pooling=False)

attention = Attention(enc_hid_dim=ENC_HID_DIM, dec_hid_dim=DEC_HID_DIM)

decoder = Decoder(output_dim=OUTPUT_DIM, emb_dim=DEC_EMB_DIM, 
                  enc_hid_dim=ENC_HID_DIM, dec_hid_dim=DEC_HID_DIM,
                  dropout_rate=DROPOUT_RATE, attention=attention)

model = Generator(encoder, decoder, device).to(device)


# In[ ]:


#model


# In[ ]:


if model.cuda:
    print(True)
else:
    print(False)


# In[ ]:


optimizer = optim.Adam(model.parameters())
pad_idx = labels['pad']

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)#ignore_index=pad_idx


# In[ ]:


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, sample in tqdm(enumerate(iterator)):
        
        src = sample['signal'].type(torch.FloatTensor).to(device)
        src = src.permute(0, 2, 1)
        trg = sample['designal'].type(torch.LongTensor).to(device)
        trg = trg.view(-1, TXT_SEQ_LEN)
       
        optimizer.zero_grad()
        
        #print('src.size - ', src.size(), ' trg.size - ', trg.size())
        #break
        #print(trg[:, 0])
        _, _, output = model(src, trg)
        #print('src.size - ', src.size(), ' trg.size - ', trg.size(), ' output.size - ', output.size())
        
        loss = criterion(output.view(-1, output.shape[2]), trg.view(-1))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# In[ ]:


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    #print('in evaluation')
    
    with torch.no_grad():
    
        for i, sample in tqdm(enumerate(iterator)):

            src = sample['signal'].type(torch.FloatTensor).to(device)
            src = src.permute(0, 2, 1)
            trg = sample['designal'].type(torch.LongTensor).to(device)
            trg = trg.view(-1, TXT_SEQ_LEN)

            _, _, output = model(src, trg, 0) #turn off teacher forcing
            #print('output - ', output)
            
            loss = criterion(output.view(-1, output.shape[2]), trg.view(-1))

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# In[ ]:


N_EPOCHS = 25
CLIP = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'msdgan.pt')

best_valid_loss = float('inf')

if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for epoch in range(N_EPOCHS):
    with torch.cuda.device(1):
        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_dataloader, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print('Saved Model.')

        print('Epoch: ', epoch+1, '| Train Loss: ', train_loss, '| Train PPL: ', math.exp(train_loss),
              '| Val. Loss: ', valid_loss, '| Val. PPL: ', math.exp(valid_loss))
    
    #print('Train loss - ', train_loss)


# In[ ]:


#torch.cuda.device(1)


# In[ ]:


# seq_len = 1100, transcript_len = 189


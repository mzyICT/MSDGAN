#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class Generator(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        #src = [batch_size, seq_len, num_feature]
        #trg = [batch_size, trg_sent_len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, encoder_h_n, hidden = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        output = trg[:, 0] #[batch_size]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[:, t] = output #[batch_size, output_dim]
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1] #[batch_size]
            output = (trg[:,t]  if teacher_force else top1)

        return encoder_outputs, encoder_h_n, outputs
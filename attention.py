#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch_size, dec hid dim]
        # encoder_outputs = [batch_size, seq_len, enc_hid_dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # hidden = [batch_size, seq_len, dec_hid_dim]
       
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, seq_len, dec_hid_dim]
        
        energy = energy.permute(0, 2, 1)
        # energy = [batch_size, dec_hid_dim, seq_len]
        
        # v = [dec_hid_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec_hid_dim]
        
        energy = torch.bmm(v, energy).squeeze(1)
        # energy = [batch_size, seq_len]
        
        return F.softmax(energy, dim=1)
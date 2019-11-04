#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout_rate, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout_rate = dropout_rate
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch_size]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [batch_size, seq_len, enc_hid_dim * 2]

        input = input.unsqueeze(1)
        # input = [batch_size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch_size, 1, emb_dim]
        
        a = self.attention(hidden, encoder_outputs)
        # a = [batch_size, seq_len]
        a = a.unsqueeze(1)
        # a = [batch_size, 1, seq_len]
        
        context = torch.bmm(a, encoder_outputs)
        # context = [batch_size, 1, enc_hid_dim * 2]
        
        gru_input = torch.cat((embedded, context), dim=2)
        # gru_input = [batch_size, 1, (enc hid dim * 2) + emb dim]
        
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # output = [batch_size, seq_len, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq_len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [batch_size, 1, dec_hid_dim]
        #hidden = [1, batch_size, dec_hid_dim]
        #this also means that output == hidden
        
        #assert (output == hidden).all()
        
        embedded = embedded.squeeze(1) #[batch_size, emb_dim]
        output = output.squeeze(1) #[batch_size, dec_hid_dim * n directions]??????????
        context = context.squeeze(1) #[batch_size, enc_hid_dim * 2]
        
        output = self.fc(torch.cat((output, context, embedded), dim=1))
        # output = [batch_size, output_dim]
        
        return output, hidden.squeeze(0)
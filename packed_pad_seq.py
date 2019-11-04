#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seqs = ['gigantic_string','tiny_str','medium_str']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(set(''.join(seqs)))


# In[18]:


vocab


# In[19]:


# make model
embed = nn.Embedding(len(vocab), 10).cuda()
lstm = nn.LSTM(10, 5).cuda()


# In[20]:


vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
vectorized_seqs


# In[21]:


# get the length of each seq in your batch
seq_lengths = torch.LongTensor([len(seq) for seq in vectorized_seqs]).cuda()
seq_lengths


# In[22]:


# dump padding everywhere, and place seqs on the left.
# NOTE: you only need a tensor as big as your longest sequence
seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long().cuda()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
	seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

seq_tensor.shape


# In[23]:


# SORT YOUR TENSORS BY LENGTH!
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]
seq_lengths, seq_tensor


# In[24]:


# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)

# embed your sequences
seq_tensor = embed(seq_tensor)

# pack them up nicely
packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())


# In[27]:


packed_input[0].shape, packed_input[1].shape


# In[29]:


#packed_input


# In[32]:


# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output, (ht, ct) = lstm(packed_input)
packed_output[0].shape, packed_output[1].shape


# In[34]:


# unpack your output if required
output, _ = pad_packed_sequence(packed_output)
print (output.shape)


# In[35]:


# Or if you just want the final hidden state?
print (ht[-1].shape)


# In[36]:


# REMEMBER: Your outputs are sorted. If you want the original ordering
# back (to compare to some gt labels) unsort them
_, unperm_idx = perm_idx.sort(0)
output = output[unperm_idx]
print (output)


# In[ ]:





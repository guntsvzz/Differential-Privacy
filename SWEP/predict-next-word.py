import torch
from transformers import AutoModelForCausalLM, \
  AutoTokenizer
# from torch import nn
import numpy as np
print("\nBegin next-word using HF GPT-2 demo ")
toker = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

seq = "Machine learning with PyTorch can do amazing"
print("\nInput sequence: ")
print(seq)

inpts = toker(seq, return_tensors="pt")
print("\nTokenized input data structure: ")
print(inpts)

inpt_ids = inpts["input_ids"]  # just IDS, no attn mask
print("\nToken IDs and their words: ")
for id in inpt_ids[0]:
  word = toker.decode(id)
  print(id, word)

with torch.no_grad():
  logits = model(**inpts).logits[:, -1, :]
print("\nAll logits for next word: ")
print(logits)
print(logits.shape)

pred_id = torch.argmax(logits).item()
print("\nPredicted token ID of next word: ")
print(pred_id)

pred_word = toker.decode(pred_id)
print("\nPredicted next word for sequence: ")
print(pred_word)

print("\nEnd demo ")


'''
Begin next-word using HF GPT-2 demo 

Input sequence: 
Machine learning with PyTorch can do amazing

Tokenized input data structure: 
{'input_ids': tensor([[37573,  4673,   351,  9485, 15884,   354,   460,   466,  4998]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}

Token IDs and their words: 
tensor(37573) Machine
tensor(4673)  learning
tensor(351)  with
tensor(9485)  Py
tensor(15884) Tor
tensor(354) ch
tensor(460)  can
tensor(466)  do
tensor(4998)  amazing

All logits for next word: 
tensor([[-114.9652, -118.0909, -123.3014,  ..., -124.5989, -127.7998,
         -118.4347]])
torch.Size([1, 50257])

Predicted token ID of next word: 
1243

Predicted next word for sequence: 
 things

End demo 
'''
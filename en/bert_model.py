from transformers import BertTokenizer, BertModel
import torch
import numpy as np


#sentences = ['she like dogs and she is beautiful and elegant', 'he like cats']

def bert_embedded(sents): #input:  batch x seq_len
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)
    model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)

    sents_tokenized = tokenizer(sents, padding=True, return_tensors="pt")
    
    print(sents_tokenized)
    sents_encodings_full = model(**sents_tokenized).last_hidden_state

    sents_encoding = []
    for sent_vecs in sents_encodings_full:
        sent_vecs =sent_vecs[1:-1].detach().numpy()
        sents_encoding.append(sent_vecs)

    return sents_encoding
           
    

if __name__ == '__main__':
    sentences = ['she like dogs and she is beautiful and elegant']
    out = bert_embedded(sentences)
    print(len(out))
    print(out[0].shape)


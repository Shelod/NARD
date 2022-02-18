import pickle
import gc

#from ch_bert_model import ch_bert_embedded
import numpy as np
import h5py
import os
import time
import torch
from nltk.corpus import wordnet as wn


torch.cuda.set_device(3)

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)
model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
'''
from transformers import AlbertTokenizer, AlbertModel
tokenizer = AlbertTokenizer.from_pretrained('albert-large-v1', do_lower_case=False)
model = AlbertModel.from_pretrained('albert-large-v1', return_dict=True)
'''
def bert_embedded(sents): #input:  batch x seq_len
    
    sents_tokenized = tokenizer(sents, padding=True, return_tensors="pt")
    
    sents_encodings_full = model(**sents_tokenized).last_hidden_state

    sents_encoding = []
    for sent_vecs in sents_encodings_full:
        sent_vecs =sent_vecs[1:-1].detach().numpy()
        sents_encoding.append(sent_vecs)

    return sents_encoding

def bert_sents(sents):
    sents_tokenized = tokenizer(sents, padding=True, return_tensors="pt")
    
    sents_encodings_full = model(**sents_tokenized).last_hidden_state

    sents_encoding = []
    for sent_vecs in sents_encodings_full:
        sent_vecs =sent_vecs[1:-1].detach().numpy()
        sent_vecs = sent_vecs.mean(axis=0)
        sents_encoding.append(sent_vecs)

    return sents_encoding

    
def embedding_dict(filename):
    data = pickle.load(open(filename,'rb'))
    definition_embedding = {}
    for synset, definitions in data.items():
        embedding = bert_sents([definitions[0]])

        definition_embedding[synset] = embedding[0]
    with open('word_embedding_mean.p','wb') as handle:
        pickle.dump(definition_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

def embedding(filename):

    data = pickle.load(open(filename,'rb'))
    definitions2embedding = {}
    count = 0
   
    for synset, definitions in data.items():

        count += 1
        definitions2embedding[synset] = []
        for definition in definitions:
            embedding = bert_embedded([definition])[0].tolist()
            embedding = pad_with_zero(embedding)
            definitions2embedding[synset].append(embedding)
                
            print('count: %s/%s' % (count, len(data)))

    with open('test_desc_data_embedding%s.p' % ('32'), 'wb') as handle:
        pickle.dump(definitions2embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def test_embedding(test_file,key):
    data = pickle.load(open(test_file, 'rb'))
    adict = {}
    seen_dict = {'stolid':'impassive.s.01', 'ecumenical':'cosmopolitan.s.03', 'mundane':'mundane.s.03', 'maternal':'maternal.a.01'}
    unseen_dict = {'twinkly':'beamish.s.01', 'fantastic':'fantastic.s.04', 'soggy':'doughy.s.01', 'used':'used.a.01','slavish':'slavish.s.02','gray':'grey.s.02','applied':'applied.a.01','unforeseen':'unanticipated.s.01','intangible':'intangible.a.02'}
    _dict = {'seen':seen_dict, 'unseen':unseen_dict}
    fix_dict = _dict[key]
    count = 0
    
    for synset, definitions in data.items():
        if synset in fix_dict.keys():
            synset = fix_dict[synset]
        else:
            synsets = wn.synsets(synset)
            flag = False
            for _synset in synsets:
                if _synset.definition().replace(' ', '').lower() == definitions[0].replace(' ', '').lower():
                    synset = _synset.name()
                    flag = True
                    break
            if not flag:
                print(synset, definitions)
        if synset not in adict.keys():
            adict[synset] = []
    
        count += 1
        for definition in definitions:
            embedding = bert_embedded([definition])[0].tolist()
            embedding = pad_with_zero(embedding)
            adict[synset].append(embedding)
                
            print('count: %s/%s' % (count, len(data)))
    with open('test_%s_data_embedding%s.p' % (key,'16all'), 'wb') as handle:
        pickle.dump(adict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

def cac(dict1,dict2):
    print(len(dict1),len(dict2))
   
    for key, value in dict1.items():

        assert key not in dict2.keys()
        dict2[key] = value
    print(1)
    with open('synset_train_data_embedding32%s.p' % ('all'), 'wb') as handle:
        pickle.dump(dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pad_with_zero(embedding, max_seq_len=32):
    '''   
    for i in range(len(embedding)):
               
        embedding[i] = list(embedding[i])
    '''
    if len(embedding)>= max_seq_len:
        embedding = embedding[:max_seq_len]
    else:
        for i in range(max_seq_len - len(embedding)):
            embedding.append([0.]*1024)
    embedding = np.array(embedding, dtype = np.float32)
    return embedding




   
if __name__ == '__main__':
    
    folder = './data/data_dict_five/'
    data_file = os.path.join(folder, 'four_dict_data-1-50477.p')
    test_seen_file = './data/en/seen_data.p'
    test_unseen_file = './data/en/unseen_data.p'
    test_desc_file = './data/en/desc_data.p'
    embedding(test_desc_file)
    #embedding_dict(data_file)
    #embedding(data_file)
    #test_embedding(test_seen_file,'seen')
    #test_embedding(test_unseen_file,'unseen')
    
    

    
    #embedding(data_file)
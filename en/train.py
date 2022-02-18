import torch
import numpy as np
from dataLoader import dataloader
from loss import LabelSmoothedCrossEntropyCriterion
import model 
import h5py
import torch.nn.functional as F 
import pickle
from nltk.corpus import wordnet as wn
from evaluate import eval_result
from torch.nn.utils.rnn import pack_padded_sequence
import os

train_name = 'en'

if train_name == 'en':
    size = 1024
else:
    size = 768

pos_dict = {'n':0, 'v':1, 's':2, 'r':3, 'a':4}
lexname_dict = pickle.load(open('/data1/cgw_data/data/lexname_index.p', 'rb'))

lr = 0.0002
seed = 1111
dropout = 0.5
batch_size = 512
embed_dim = size     #fixed
cuda_able = True
hidden_size = size
bidirectional = True
weight_decay = 0.001
attention_size = size  
pos_size = 5
lexname_size = 45
#sequence_length = 16
output_size = size   #fixed

save_path = './RD_model.pt'

use_cuda = torch.cuda.is_available() and cuda_able

torch.cuda.set_device(0)
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_matrix(synset_embedding, target_synset):
    synset_embedding_matrix = []

    for synset in target_synset:
        synset_embedding_matrix.append(synset_embedding[synset])

    synset_embedding_matrix = torch.from_numpy(np.array(synset_embedding_matrix)).float()
    print('target synset matrix shape is:', synset_embedding_matrix.shape)
    return synset_embedding_matrix

setup_seed(seed) 
print(use_cuda)

import time
from tqdm import tqdm

if train_name == 'en':

    train_file_index_path = '/data1/cgw_data/data/four_dict_data-1-50477.p'
    #train_file = './synset_train_definition_embedding data_all.p'
    print('loading training data')
    train_data_index = dataloader(data_file=train_file_index_path,save_path='/data1/cgw_data/train_data/sorted_train_data16_with_pad.hdf5',batch_size=batch_size, save=False, remove_item=[0])
    #train_data = pickle.load(open(train_file, 'rb'))
    print('loading training data finished')
    synset_embedding_file = './synset_embeddings/synset_embedding_lmms2048_with_reduce_dim.p'
    #synset_embedding_file ='./synset_embeddings/synset_embedding_ares2048_to1024.p'
    #synset_embedding_file = './synset_embeddings/synset_embedding_sbert_normalized.p'
    #synset_embedding_file = './synset_embeddings/bert_embedding_mean_normalized.p'
    synset_embedding = pickle.load(open(synset_embedding_file, 'rb'))
    
    test_seen_file = '/data1/cgw_data/test_data/test_seen_data_embedding16all.p'
    test_seen_data = pickle.load(open(test_seen_file, 'rb'))

    test_desc_file = '/data1/cgw_data/test_data/test_desc_data_embedding16.p'
    test_desc_data = pickle.load(open(test_desc_file, 'rb'))

    test_unseen_file = '/data1/cgw_data/test_data/test_unseen_data_embedding16all.p'
    test_unseen_data = pickle.load(open(test_unseen_file, 'rb'))
    test_data_all = pickle.load(open('/data1/cgw_data/train_data/synset_train_definition_embedding data_all.p', 'rb'))
    analyses_sense_number = pickle.load(open('./analyses/sense_number_synsets.p', 'rb'))
    analyses_sense_count = pickle.load(open('./analyses/sense_count_synsets.p', 'rb'))


    target_synset = pickle.load(open(train_file_index_path, 'rb')).keys()

    synset_embedding_matrix = get_matrix(synset_embedding, target_synset)

else:
    train_file_index_path = './data/cn/train_data.p'
    train_data_index = dataloader(data_file=train_file_index_path,save_path='./ch_sorted_train_data32_with_pad.hdf5', batch_size=batch_size, save=False, remove_item=[])

    synset_embedding_file = './ch_word_embedding.p'
    synset_embedding = pickle.load(open(synset_embedding_file, 'rb'))

    test_file = './ch_test_desc_data_embedding32.p'
    test_data = pickle.load(open(test_file, 'rb'))

    target_synset = pickle.load(open(train_file_index_path, 'rb')).keys()

    synset_embedding_matrix = get_matrix(synset_embedding, target_synset)


print('index file load finished')
#train_data, val_data = read_part()
print('data file load finished')

lstm_attn = model.bilstm_attn(batch_size=batch_size,
                                output_size=output_size,
                                hidden_size=hidden_size,
                                embed_dim=embed_dim,
                                pos_size = pos_size,
                                lexname_size = lexname_size,
                                bidirectional=bidirectional,
                                dropout=dropout,
                                use_cuda = use_cuda,
                                attention_size=attention_size,
                                out_embedding_matrix=synset_embedding_matrix)

#lstm_attn = torch.nn.DataParallel(lstm_attn)
if use_cuda:
    lstm_attn = lstm_attn.cuda()

#val_data = dataloader(data_file=val_file_path, batch_size=1)
#data 和label的数据类型是 float32

train_loss = []
eval_loss = []

epochs = 2

optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=lr, weight_decay=weight_decay)
#criterion = TripletCosineLoss(margin = 0.6, batch_size=batch_size)
criterion = LabelSmoothedCrossEntropyCriterion()
#criterion = torch.nn.CrossEntropyLoss()

def analyses(test_data, sense_number, sense_count):
    lstm_attn.eval()
    index = 0
    
    out_dict = {1:0,2:0,3:0,4:0,5:0,6:0}
    all_dict = {1:0,2:0,3:0,4:0,5:0,6:0}

    out_dict_count = {1:0,2:0,3:0,4:0,5:0,6:0}
    all_dict_count = {1:0,2:0,3:0,4:0,5:0,6:0}

    with torch.no_grad():
        for label, data in tqdm(test_data.items(), desc='analysing'):
            for vecs in data:
                index += 1
                count = 0
                for vec in vecs:
                    count += 1
                    if all(vec == np.zeros((size,), dtype=np.float32)):
                        count -= 1
                        break
        
                definition = torch.from_numpy(vecs)   
                definition = torch.unsqueeze(definition, 0)  
                definition = definition.cuda()
        
                out_vector, out_vector2, out_vector3 = lstm_attn(definition, torch.tensor([count]))
                out_vector = out_vector.cpu()
                
                out_vector = torch.squeeze(out_vector)

                true_label = list(target_synset).index(label)
                result1, result10, result100, pred_rank = eval_result(out_vector, true_label,desc=False)
        
                for key in sense_number.keys():
                    if label in sense_number[key]:
                        all_dict[key] += 1
                        out_dict[key] += result10
                    if label in sense_count[key]:
                        all_dict_count[key] += 1
                        out_dict_count[key] += 1
                

                
    print('analysing sense number------------------')
    print(out_dict)
    print(all_dict)

    print('analysing sense number------------------')
    print(out_dict_count)
    print(all_dict_count)


            

def evaluate(test_data, desc): #2092 
    lstm_attn.eval()
    
    with torch.no_grad():
        index = 0
        correct1, correct10, correct100 = 0, 0, 0
        pred_ranks = []
        
        for label, data in tqdm(test_data.items(), desc='test seen Processing'):
            
            for vecs in data:
                index += 1
                count = 0
                for vec in vecs:
                    count += 1
                    if all(vec == np.zeros((size,), dtype=np.float32)):
                        count -= 1
                        break
        
                definition = torch.from_numpy(vecs)   
                definition = torch.unsqueeze(definition, 0)  
                definition = definition.cuda()
        
                out_vector, out_vector2, out_vector3 = lstm_attn(definition, torch.tensor([count]))
                out_vector = out_vector.cpu()
                
                out_vector = torch.squeeze(out_vector)

                if desc:
                    label_synsets = [synset.name() for synset in wn.synsets(label)]
                    true_label = [list(target_synset).index(_label) for _label in label_synsets]
                else:
                    true_label = list(target_synset).index(label)
        
                result1, result10, result100,pred_rank = eval_result(out_vector, true_label, desc)
                correct1 += result1
                correct10 += result10
                correct100 += result100
                pred_ranks.append(pred_rank)
           
    print(index, correct1, correct10, correct100, np.median(pred_ranks), np.sqrt(np.var(pred_ranks)))
                
    return correct1

def train():
    lstm_attn.train()
    total_loss = 0
    for vectors, synsets,lengths,poses,lexnames in tqdm(train_data_index, desc='Train Processing'):
        optimizer.zero_grad()

        target, pos_target, lexname_target = lstm_attn(vectors,lengths)

        loss = criterion(target, synsets)
        loss2 = criterion(pos_target, poses)
        loss3 = criterion(lexname_target,lexnames)
        
        loss = loss +loss2 +loss3
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss

print('begin training ...')
total_start_time = time.time()
high = 0
try:
    print('-' * 90)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        loss = train()
        
        #plot_loss_total += loss
        train_loss.append(loss*1000.)
        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                              time.time() - epoch_start_time,
                                                                      loss))
        loss = evaluate(test_seen_data, desc=False)
        #loss = evaluate(test_unseen_data, desc=False)
        loss = evaluate(test_desc_data, desc=True)
        #analyses(test_data_all, analyses_sense_number, analyses_sense_count)
        eval_loss.append(loss*1000.)
        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                              time.time() - epoch_start_time,
                                                                              loss))
        if loss > high:
            high = loss
            torch.save(lstm_attn.state_dict(), "./lstm_attn.pth")

        
except KeyboardInterrupt:
   
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))        


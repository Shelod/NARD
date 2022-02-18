import torch
import pickle
import numpy as np
import model
from tqdm import tqdm

from evaluate import eval_result

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)
Model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)


size = 1024
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
dropout = 0.5
use_cuda = torch.cuda.is_available() and cuda_able


def get_matrix(synset_embedding, target_synset):
    synset_embedding_matrix = []

    for synset in target_synset:
        synset_embedding_matrix.append(synset_embedding[synset])

    synset_embedding_matrix = torch.from_numpy(np.array(synset_embedding_matrix)).float()
    print('target synset matrix shape is:', synset_embedding_matrix.shape)
    return synset_embedding_matrix

train_file_index_path = '../en/data/four_dict_data-1-50477.p'
target_synset = pickle.load(open(train_file_index_path, 'rb')).keys()

synset_embedding_file = '../en/synset_embeddings/synset_embedding_lmms2048_with_reduce_dim.p'
    
synset_embedding = pickle.load(open(synset_embedding_file, 'rb'))

synset_embedding_matrix = get_matrix(synset_embedding, target_synset)

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



lstm_attn.load_state_dict(torch.load('./lstm_attn.pth'))
lstm_attn = lstm_attn.cuda()
def analyses(test_data, sense_number, sense_count, target_synset):
    lstm_attn.eval()
    index = 0
    
    out_dict = {1:0,2:0,3:0,4:0,5:0,6:0}
    all_dict = {1:0,2:0,3:0,4:0,5:0,6:0}

    out_dict_count = {1:0,2:0,3:0,4:0,5:0,6:0}
    all_dict_count = {1:0,2:0,3:0,4:0,5:0,6:0}

    with torch.no_grad():
        for label, data in tqdm(test_data.items(), desc='analysing'):
            for vecs in data[:1]:
                index += 1
                count = 0
                for vec in vecs:
                    count += 1
                    if all(vec == np.zeros((1024,), dtype=np.float32)):
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
                        out_dict_count[key] += result10
                

                
    print('analysing sense number------------------')
    print(out_dict)
    print(all_dict)

    print('analysing sense number------------------')
    print(out_dict_count)
    print(all_dict_count)

def bert_embedded(sents): #input:  batch x seq_len
    
    sents_tokenized = tokenizer(sents, padding=True, return_tensors="pt")
    
    sents_encodings_full = Model(**sents_tokenized).last_hidden_state

    sents_encoding = []
    for sent_vecs in sents_encodings_full:
        sent_vecs =sent_vecs[1:-1].detach().numpy()
        sents_encoding.append(sent_vecs)

    return sents_encoding

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

def single_sentence(sentence):
    count = 0

    vecs = bert_embedded(sentence)[0]
    for vec in vecs:
        count += 1
        if all(vec == np.zeros((1024,), dtype=np.float32)):
            count -= 1
            break
        
    definition = torch.from_numpy(vecs)   
    definition = torch.unsqueeze(definition, 0)  
    definition = definition.cuda()
        
    out_vector, out_vector2, out_vector3 = lstm_attn(definition, torch.tensor([count]))
    out_vector = out_vector.cpu()
                
    out_vector = torch.squeeze(out_vector)

    _, top10_indices = torch.topk(out_vector, 10)

    for label in top10_indices:
        print(list(target_synset)[label])
    

if __name__ == '__main__':
    '''
    analyses_sense_number = pickle.load(open('./sense_number_synsets.p', 'rb'))
    analyses_sense_count = pickle.load(open('./sense_count_synsets.p', 'rb'))
    test_data_all = pickle.load(open('../en/train_data/synset_train_definition_embedding data_all.p', 'rb'))

    analyses(test_data_all, analyses_sense_number, analyses_sense_count,target_synset)
    '''
    sentences = ['put up with something or someone unpleasant']
    single_sentence(sentences)

import os
import json
import pickle


def load(folder):
    
    for name in ['desc.json', 'dev.json', 'seen.json', 'train.json', 'unseen.json']:
        path = os.path.join(folder, name)
        _dict = {}
        
        with open(path, 'r') as f:
            data = json.load(f)
            for d in data:
                word = d['word'].lower()
                definition = d['sememes']
                if word in _dict.keys():
                    pass
                else:
                    _dict[word] = definition
        print(len(_dict))
        with open(os.path.join(folder,name.split('.')[0]+'_sememes.p'), 'wb') as handle:
            pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_word(folder):
    word2index = {}
    index2word = {}
    index2word[0] = 'UNKNOW'
    word2index['UNKNOW'] = 0
    with open(os.path.join(folder, 'target_words.txt'), 'r') as f:
        count = 0
        for line in f:
            count += 1
            word = line.strip()
            word2index[word] = count
            index2word[count] = word
    
    with open('ch_targetWord2index.p', 'wb') as handle:
        pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('ch_index2targetWord.p', 'wb') as handle:
        pickle.dump(index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_cn_data(folder):
    for name in ['desc.json', 'dev.json', 'question.json', 'seen_test.json', 'train.json',
                     'unseen_test.json']:
        path = os.path.join(folder, name)
        _dict = {}
        with open(path, 'r') as f:
            data = json.load(f)
            for d in data:
                word = d['word']
                definition = d['definition']

                if word in _dict.keys():
                    _dict[word].append(definition)
                else:
                    _dict[word] = [definition]

    print(len(_dict))
    with open(os.path.join(folder,name.split('.')[0]+'_data.p'), 'wb') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    folder = './data/en'
    load(folder)   
        
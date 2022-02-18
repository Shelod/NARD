from nltk.corpus import wordnet as wn
import pickle

def get_charater_from_synset(synset_name, charater):
    synset = wn.synset(synset_name)
    lemmas = synset.lemmas()
    alist = []
    for lemma in lemmas:
        word = lemma.name()
        if word in charater.keys():
            alist += charater[word]
        else:
            pass
    if not alist:
        return 'none'
    count = 0
    count_idx = 0
    for i in range(len(alist)):
        if alist.count(alist[i]) > count:
            count = alist.count(alist[i])
            count_idx = i
    
    return alist[count_idx]


if __name__ == '__main__':
    
    _dict = {}
    _dict['none'] = 0
    count = 1
    with open('./babelnet/rootaffix_all.txt', 'r') as f:
        for line in f:
            root = line.strip()
            _dict[root] = count
    print(len(_dict))
    with open('root_affix_to_index.p', 'wb') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    '''
    root_affix = pickle.load(open('./data/en/train_root_affix.p', 'rb'))

    train_data = pickle.load(open('./data/data_dict_five/four_dict_data-1-50477.p', 'rb'))
    root_affix_for_synset = {}
    for synset, definitions in train_data.items():
        root_affix_for_synset[synset] = get_charater_from_synset(synset, root_affix)
    print(len(root_affix_for_synset))
    with open('./synset_root_affix.p', 'wb') as handle:
        pickle.dump(root_affix_for_synset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
        


from nltk.corpus import wordnet as wn
import pickle
import numpy as np


if __name__ == '__main__':
    '''
    _dict = {1:[],2:[],3:[],4:[],5:[],6:[]}
    synset_target = list(pickle.load(open('../en/data/four_dict_data-1-50477.p', 'rb')).keys())

    word_file = '../en/data/target_words.txt'
    with open(word_file, 'r') as f:
        for line in f:
            word = line.strip()
            synsets = wn.synsets(word)
            sense_number = min(len(synsets),6)
            for synset in synsets:
                if synset.name() not in synset_target:
                    continue
                if synset.name() not in _dict[sense_number]:
                    _dict[sense_number].append(synset.name())

    with open('sense_number_synsets.p', 'wb') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''

    _dict = {1:[],2:[],3:[],4:[],5:[],6:[]}
    synset_target = list(pickle.load(open('../en/data/four_dict_data-1-50477.p', 'rb')).keys())
    
    word_file = '../en/data/target_words.txt'
    count_list = []
    word_list = []
    index2word = {}
    index = 0
    with open(word_file, 'r') as f:
        for line in f:
            
            word = line.strip()
            index2word[index] = word
            word_list.append(index)
            index += 1
            lemmas = wn.lemmas(word)
            count = 0
            for lemma in lemmas:
                count += lemma.count()
            
            count_list.append(count)

    count_list = np.array(count_list)
    word_list = np.array(word_list)
    indices = np.argsort(count_list)
    word_list = word_list[indices]
    count = 0
    for word in word_list:
        count += 1
        if count<5000:
            index = 1
        elif count<10000:
            index = 2
        elif count<20000:
            index = 3
        elif count<30000:
            index = 4
        elif count<40000:
            index = 5
        else:
            index = 6
        word = index2word[word]
        synsets = wn.synsets(word)
        for synset in synsets:
            if synset.name() not in synset_target:
                continue
            if synset.name() not in _dict[index]:
                _dict[index].append(synset.name())
        with open('sense_count_synsets.p', 'wb') as handle:
            pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        


    



            

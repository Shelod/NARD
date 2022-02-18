import numpy as np
import os
import h5py
import pickle
import gc
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import time
from nltk.corpus import wordnet as wn


pos_dict = {'n':0, 'v':1, 's':2, 'r':3, 'a':4}
lexname_dict = pickle.load(open('/data1/datasets/hxb_data/en/lexname_index.p','rb'))

class dataloader():

    def __init__(self, data_file,save_path, batch_size, save=True, remove_item=[]):
        super(dataloader,self).__init__()
        self.remove_item = remove_item
        self.save_path = save_path
    
        if save:
            self.train_data = pickle.load(open(data_file, 'rb'))
            self.pairs, self.labels, self.lengths, self.poses, self.lexnames, = self.data_handle()
            self._shuffle()
            self._save()
        else:
            if self.remove_item != []:
                h5 = h5py.File('/data1/cgw_data/train_data/sorted_train_unseen_data16_with_padNew.hdf5', 'r')
            else:
    
                h5 = h5py.File(self.save_path, 'r')
                
            self.pairs = np.array(h5['index'])
            
            self.labels = np.array(h5['label'])
            self.lengths = np.array(h5['length'])
           
            self.poses = np.array(h5['pos'])
           
            self.lexnames = np.array(h5['lexname'])



        self.batch_size = batch_size
        self.step = 0
        self.stop = len(self.pairs) // self.batch_size
        
            
    def __iter__(self):
        return self

    
    def data_handle(self):

        pairs = []
        labels = []
        lengths = []
        poses = []
        lexnames = []
        final= 0
    
        index = 0
        
        for synset in self.train_data.keys():
            
            definitions = self.train_data[synset]
            wn_synset = wn.synset(synset)
            pos = pos_dict[wn_synset.pos()]
            lexname = lexname_dict[wn_synset.lexname()]
            count = 0
            for definition in definitions:
                
                if (synset,count) in self.remove_item:
                    final += 1
                    print('final:',final)
                    continue
                h = 0
                for vec in definition:
                    h += 1
                    if all(vec == np.zeros((1024,), dtype=np.float32)):
                        h -= 1
                        break
            
                lengths.append(h)
                poses.append(pos)
                lexnames.append(lexname)
                pairs.append(definition)
                labels.append(index)
                count += 1
            index += 1
            print(index)

        pairs = np.array(pairs)
        print(pairs.shape)
        labels = np.vstack(labels)
        print(labels.shape)
        lengths = np.vstack(lengths)
        lexnames = np.vstack(lexnames)
        poses = np.vstack(poses)

        return pairs,labels,lengths,poses,lexnames
    
    def _shuffle(self):
        indices = np.arange(self.pairs.shape[0])
        np.random.shuffle(indices)
        self.lengths = self.lengths[indices]

        self.pairs = self.pairs[indices]
        self.labels = self.labels[indices]
        self.poses = self.poses[indices]
        self.lexnames = self.lexnames[indices]

    def _save(self):
        h5file = h5py.File(self.save_path, 'w')
        print(self.pairs.shape)
        dset1 = h5file.create_dataset('index', data = self.pairs)
        dset2 = h5file.create_dataset('label', data = self.labels)
        dset3 = h5file.create_dataset('length', data = self.lengths)
        dset4 = h5file.create_dataset('pos', data = self.poses)
        dset5 = h5file.create_dataset('lexname', data = self.lexnames)

        h5file.close()

    def __next__(self):

        if self.step >= self.stop:
            #self._shuffle()
            
            self.step = 0
            raise StopIteration()
        out = self.pairs[self.step*self.batch_size:self.step*self.batch_size+self.batch_size]
        label = self.labels[self.step*self.batch_size:self.step*self.batch_size+self.batch_size]
        length = self.lengths[self.step*self.batch_size:self.step*self.batch_size+self.batch_size]
        pos = self.poses[self.step*self.batch_size:self.step*self.batch_size+self.batch_size]
        lexname = self.lexnames[self.step*self.batch_size:self.step*self.batch_size+self.batch_size]

        label = torch.from_numpy(label.flatten()).cuda()
    
        out = torch.from_numpy(out)
        out = out.float()
        out = out.cuda()
        
        length = torch.from_numpy(length.flatten()).cuda()
        pos = torch.from_numpy(pos.flatten()).cuda()
        lexname = torch.from_numpy(lexname.flatten()).cuda()

        sorted_seq_lengths, indices = torch.sort(length, descending=True)
        out = out[indices]
        label = label[indices]
        lexname = lexname[indices]
        pos = pos[indices]
        '''
        for i in range(self.batch_size):
            if sorted_seq_lengths[i] == 16:
                continue
            else:
                sorted_seq_lengths[i] = sorted_seq_lengths[i]-1
        '''
        
        self.step += 1
        return out, label,sorted_seq_lengths,pos, lexname

        

if __name__ == '__main__':


    
    from nltk.corpus import wordnet as wn

    unseen_data = pickle.load(open('./data/four_dict_dataunseen.p', 'rb'))

    sorted_train_data = pickle.load(open('./data/four_dict_data-1-50477.p', 'rb'))

    remove_list = []
    index = 0

    for synset, definitions in unseen_data.items():
        
        if synset not in sorted_train_data.keys():
            
            print('index:',index)
            index += 1
            continue
        
        for definition in definitions:
            
            if definition not in sorted_train_data[synset]:
                continue
            
            #print('count:',count)
            count = sorted_train_data[synset].index(definition)
            remove_list.append((synset, count))
        

    print(len(remove_list))

    data_file = './train_data/synset_train_definition_embedding data_all.p'
    save_path = './train_data/sorted_train_unseen_data16_with_padNew.hdf5'
    
    train_data = dataloader(data_file=data_file, save_path=save_path, batch_size=16,remove_item=remove_list)

    
   
    



        

        



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence,pad_sequence
#from transformer_encoder import EncoderTransformer

#from torch.nn import LayerNorm, ModuleList
#from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class bilstm_attn(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embed_dim, pos_size, lexname_size,bidirectional, dropout, use_cuda, attention_size, out_embedding_matrix):
                 
        super(bilstm_attn, self).__init__()
        
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.pos_size = pos_size
        self.lexname_size = lexname_size
        self.dropout = dropout
        
        self.use_cuda = use_cuda
        self.bidirectional = bidirectional
        self.out_embedding_matrix = out_embedding_matrix
        self.dropout = dropout
    
        self.layer_size = 2
        
        self.lstm = nn.LSTM(self.embed_dim, 
                            self.hidden_size,
                            self.layer_size,
                            batch_first=True,
                            dropout = self.dropout,
                            bidirectional=bidirectional)#.flatten_parameters()

        #self.transformer = EncoderTransformer()
        self.attention_query = nn.Linear(self.hidden_size * 2, self.attention_size)
        self.attention_key = nn.Linear(self.hidden_size * 2, self.attention_size)
        self.attention_value = nn.Linear(self.hidden_size * 2, self.attention_size)
        self.scale_factor = np.sqrt(self.attention_size)
        
        self.embedding_dropout = nn.Dropout(self.dropout)
    
        #self.label = nn.Linear(self.hidden_size * 2, self.output_size)
        self.decoder = nn.Linear(self.output_size, self.out_embedding_matrix.size(0), bias=False)

        self.pos_decoder = nn.Linear(self.output_size, self.pos_size)
        self.lexname_decoder = nn.Linear(self.output_size, self.lexname_size)
       
        self.decoder_bias = nn.Linear(self.output_size, 1, bias=False)
           
        #self.label = nn.Linear(hidden_size, output_size)
        self.label = nn.Linear(hidden_size * 2 + attention_size, output_size)
        
        self.init_weigths()
        

    def init_weigths(self):
        initrange = 0.1

        self.attention_key.bias.data.fill_(0)
        self.attention_key.weight.data.uniform_(-initrange, initrange)
        self.attention_query.bias.data.fill_(0)
        self.attention_query.weight.data.uniform_(-initrange, initrange)
        self.attention_value.bias.data.fill_(0)
        self.attention_value.weight.data.uniform_(-initrange, initrange)
        
        self.label.bias.data.fill_(0)
        self.label.weight.data.uniform_(-initrange, initrange)
        
        self.decoder.weight = nn.Parameter(self.out_embedding_matrix)
        self.decoder.weight.requires_grad = False
        
        #self.decoder_bias.bias.data.fill_(0)
        self.decoder_bias.weight.data.uniform_(-initrange, initrange)

        self.pos_decoder.bias.data.fill_(0)
        self.pos_decoder.weight.data.uniform_(-initrange, initrange)

        self.lexname_decoder.bias.data.fill_(0)
        self.lexname_decoder.weight.data.uniform_(-initrange, initrange)


    def attention_net(self, lstm_output):
        
        lstm_output = lstm_output.permute(1,0,2)
        
        q = self.attention_query(lstm_output)
        k = self.attention_key(lstm_output)
        v = self.attention_value(lstm_output)

        u = torch.bmm(q.permute(1,0,2), k.permute(1,2,0))

        u = u / self.scale_factor
        a = F.softmax(u, 2)
        #bsz*N*N * bsz*N*n_hiddenXnD -> bsz*N*n_hiddenXnD -> N*bsz*n_hiddenXnD
        c = torch.bmm(a, v.permute(1,0,2)).permute(1,0,2)
        output = torch.cat([lstm_output, c], 2) #N*bsz*n_hiddenXnDX2

        output = output.permute(1,0,2)
        
        output = torch.sum(output, dim=1)
       
        return output
        
    def forward(self, inputs, lens):

        #input = input.permute(1, 0, 2)
        
        #input = self.embedding_dropout(input)
        '''
        pad_mask = [torch.ones(x) for x in lens] + [torch.ones(32)]
        pad_mask = pad_sequence(pad_mask, batch_first=True, padding_value=0)[:-1]

        output = self.transformer(inputs, pad_mask)
        output = torch.sum(output, dim=1)
        
        #print(input.shape)
        '''
        #inputs = self.embedding_dropout(inputs)
        embedded = pack_padded_sequence(input=inputs, lengths=lens, batch_first=True)
        
        lstm_output, _= self.lstm(embedded)
        
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        lstm_output = self.embedding_dropout(lstm_output)

        output = self.attention_net(lstm_output)

        #output = self.embedding_dropout(output)
        
        logits = self.label(output)

        decoded = self.decoder(logits)

        #pos:
        poslogits = self.pos_decoder(logits)

        #lexname:
        lexnamelogits = self.lexname_decoder(logits)

        bias = self.decoder_bias(self.decoder.weight).squeeze(-1).unsqueeze(0)
        
        return decoded +bias  , poslogits, lexnamelogits#, rootlogits

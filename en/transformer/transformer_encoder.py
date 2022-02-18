from torch.nn import Module, LayerNorm, ModuleList, Dropout
from onmt.encoders.transformer import TransformerEncoderLayer
from positionalEncoding import PositionalEncoding
import math


pad_token_index = 0
class EncoderTransformer(Module):

    def __init__(self):
        super().__init__()

        self.encoder_transformer_heads = 8
        self.encoder_transformer_hidden_size = 1024
        self.encoder_transformer_layers = 6
        self.encoder_transformer_dropout = 0.1
        self.encoder_transformer_positional_encoding = False
        self.encoder_transformer_scale_embeddings = True
        self.resulting_embeddings_size = 1024


        if self.encoder_transformer_positional_encoding:
            self.positional_encoding = PositionalEncoding(self.resulting_embeddings_size)
            # self.add_module("pe", self.positional_encoding)
        else:
            self.positional_encoding = None

        if self.encoder_transformer_scale_embeddings:
            self.embeddings_scale = math.sqrt(float(self.resulting_embeddings_size))
        else:
            self.embeddings_scale = None

        self.dropout = Dropout(self.encoder_transformer_dropout)

        self.transformer = ModuleList([TransformerEncoderLayer(d_model=self.resulting_embeddings_size,
                                                               heads=self.encoder_transformer_heads,
                                                               d_ff=self.encoder_transformer_hidden_size,
                                                               dropout=self.encoder_transformer_dropout,
                                                               attention_dropout=self.encoder_transformer_dropout)
                                       for _ in range(self.encoder_transformer_layers)])
        self.layer_norm = LayerNorm(self.resulting_embeddings_size, eps=1e-6)

        self.encoder_output_size = self.resulting_embeddings_size

    # input:
    #   - embeddings     List[FloatTensor] - features x batch x seq x hidden
    #   - pad_mask       LongTensor        - batch x seq
    # output:
    #   - output         FloatTensor       - batch x seq x hidden
    def forward(self, embeddings, pad_mask):
        if self.embeddings_scale is not None:
            embeddings = embeddings * self.embeddings_scale
        if self.positional_encoding is not None:
            embeddings = embeddings + self.positional_encoding(embeddings.size(1))
        embeddings = self.dropout(embeddings)
        pad_mask = pad_mask.eq(pad_token_index).unsqueeze(1).cuda()  # batch x 1 x seq
        for layer in self.transformer:
            embeddings = layer(embeddings, pad_mask)
        embeddings = self.layer_norm(embeddings)
        return embeddings
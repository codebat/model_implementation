import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size//self.heads

        assert(self.head_dim*self.heads==self.embed_size), "heads splitting is not consistent with embed size"

        self.query = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim,bias=False)

        self.fc_layer = nn.Linear(self.embed_size,self.embed_size,bias=False)
    
    def forward(self, key, query, value, mask):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        values = value.reshape(N,value_len,self.heads, self.head_dim)
        querys = query.reshape(N,query_len,self.heads,self.head_dim)
        keys = key.reshape(N,key_len,self.heads,self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        querys = self.query(querys)

        #multiply queries and keys using einsum 
        weights = torch.einsum("nqhd,nkhd->nhqk", querys, keys)

        if mask is not None:
            weights.masked_fill(mask==0, float('-1e20'))
        
        attention = torch.softmax(weights/(self.embed_size**0.5),dim=3)

        out = torch.einsum("nhqk,nkhd->nqhd",attention,values).reshape(N,query_len,self.embed_size)

        out = self.fc_layer(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, mask):
        attention = self.attention(key, query, value, mask)

        x = self.dropout(self.norm1(query+attention))

        ff = self.feed_forward(x)

        out = self.dropout(self.norm2(ff+x))

        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.positional_embedding = nn.Embedding(max_length,embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N,seq_len = x.shape

        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        out = self.dropout(self.positional_embedding(positions)+self.word_embedding(x))

        for layer in self.layers:
            layer(out, out, out, mask)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, device, dropout):
        super(DecoderBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, tgt_mask):
        attention = self.attention(x,x,x,tgt_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(key, query, value, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size,embed_size)
        self.positional_embedding = nn.Embedding(max_length,embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, device, dropout) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_len = x.shape

        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x)+self.positional_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)
        
        out = self.fc_out(x)
        return out

class Transformers(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size = 256,
        num_layers = 6,
        forward_expansion = 4,
        heads = 8,
        dropout = 0,
        device = 'cuda',
        max_length = 100
    ):
        super(Transformers,self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        N,tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones(tgt_len,tgt_len)).expand(N,1,tgt_len,tgt_len)
        return tgt_mask.to(self.device)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src,src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor([[1,3,4,2,5,7,6,8,9],[1,4,3,5,6,0,8,9,2]]).to(device)

    tgt = torch.tensor([[1,3,2,4,5,6],[3,4,2,5,0,9]]).to(device)

    src_pad_idx = 0
    tgt_pad_idx = 0
    src_vocab_size = 10
    tgt_vocab_size = 8
    model = Transformers(src_vocab_size,tgt_vocab_size,src_pad_idx,tgt_pad_idx).to(device)
    out = model(x, tgt[:,:-1])
    print(out,out.shape)



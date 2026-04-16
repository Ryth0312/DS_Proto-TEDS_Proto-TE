#%% Import 
'''
The encoder modules for EvoPNet
'''
import sys
sys.path.append('..')
from Configs import *
import torch, math, copy
import torch.nn.init as init
import numpy as np
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
#from Layers.Utils import *


#%% Utility modules

class PositionalEncoding(nn.Module):
    ''' Sinusoidal positional encoding for Transformer '''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model) for (T, N, D) format
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T, N, D)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    ''' Position-wise Feed-Forward Network '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x)))), None


def generate_causal_mask(sz):
    ''' Generate a causal (upper-triangular) mask for Transformer '''
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask


# %% Variable-length input RNN
class CustomRNN(nn.LSTM):
    """ https://blog.csdn.net/winter2121/article/details/115239554 """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, lengths):
       
        package = nn.utils.rnn.pack_padded_sequence(input, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        result, hn = super().forward(package)
        output, lens = nn.utils.rnn.pad_packed_sequence(result, batch_first=self.batch_first, total_length=input.shape[self.batch_first])
        return output, hn
    
         
class TemporalEncoder(nn.Module):
    ''' Generate a temporal encoder for all features'''
    def __init__(self, input_size, t_emb=128, p=0.5, num_layers=2):
        super(TemporalEncoder, self).__init__()
        
        #Choose the sequential embedding
        self.p = p
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=self.p)
        # Define Encoder
        self.rnn = CustomRNN(input_size, t_emb, batch_first=True, dropout=self.p, num_layers=2)
        
    def forward(self, seq, lens):
        x, (hn, cn) = self.rnn(seq, lens)
        all_ht, encoded_x = x, hn[-1]
        all_ht, encoded_x = self.dropout(all_ht), self.dropout(encoded_x)
        return all_ht, encoded_x

    def _init_weights(self):
        # print(self.modules())
        print('Initialize weights')
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)
                #print(m.weight)  


#%% Channel-wise Temporal Encoder
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    
class ChannelwiseTemporalEncoder(nn.Module):
    ''' Generate a temporal encoder for each channel'''
    def __init__(self, input_size, hidden_size, p=0.3, num_layers=2):
        super(ChannelwiseTemporalEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p = p
        
        self.dropout = nn.Dropout(p)
        self.channel_encoder = CustomRNN(1, self.hidden_size, batch_first=True, dropout=self.p, num_layers=self.num_layers)
        self.encoders = clones(self.channel_encoder, self.input_size)
        
    def forward(self, x, lens):
        
        # 验证输入形状和维度
        if len(x.shape) != 3 or x.shape[-1] == 0:
            raise ValueError("输入x的形状应为(batch_size, sequence_length, feature_dim)，且feature_dim不能为0")
        
        feature_dim = x.shape[-1]
        if feature_dim > len(self.encoders):
            raise ValueError("特征维度大于编码器数量")
        
        # Concat channel-wise embeddings
        all_hts, (embeds, _) = self.encoders[0](x[:,:,0].unsqueeze(-1), lens)
        embeds = embeds[-1]
        for i in range(1,feature_dim):
            channel_all_ht, (channel_embed, _) = self.encoders[i](x[:,:,i].unsqueeze(-1), lens) # b 1 h
            embeds = torch.cat((embeds, channel_embed[-1]), axis=-1) # b (i+1)*h
            all_hts = torch.cat((all_hts, channel_all_ht), axis=-1) # b t (i+1)*h
        
        # Dropout
        all_hts = self.dropout(all_hts)
        embeds = self.dropout(embeds)
        
        return all_hts, embeds


#%% Prediction Model (Encoder wrapper for reconstruction pretraining)
class PredictionModel(nn.Module):
    ''' Wraps a ChannelwiseTemporalEncoder for reconstruction-based pretraining.
        Each channel's hidden states are projected back to 1-dim via a linear layer,
        then concatenated to reconstruct the original input shape.
    '''
    def __init__(self, encoder, input_size, hidden_size, num_classes=1):
        super(PredictionModel, self).__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.hidden_size = hidden_size
        # One linear projection per channel: hidden_size -> 1
        self.channel_projections = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(input_size)
        ])

    def forward(self, x, lens):
        all_hts, embeds = self.encoder(x, lens)
        # all_hts: (batch, seq_len, input_size * hidden_size)
        # Split into per-channel hidden states and project each back to dim 1
        reconstructed = []
        for i in range(self.input_size):
            # Extract channel i's hidden states
            channel_ht = all_hts[:, :, i * self.hidden_size:(i + 1) * self.hidden_size]
            # Project back to 1-dim
            channel_recon = self.channel_projections[i](channel_ht)  # (batch, seq_len, 1)
            reconstructed.append(channel_recon)
        # Concatenate all channels: (batch, seq_len, input_size)
        all_hts_proj = torch.cat(reconstructed, dim=-1)
        return all_hts_proj, embeds


#%% Transformer-based Encoder
class TFEncoder(nn.Module):
    def __init__(self, feat_size, embed_size, nhead=4, nhid=256, num_layers=2,
                 p=0.3):  # embed_size 特征维度 200, nhead 模型head数量 2, nhid 前向网络的维数 200, nlayers Encoder的层数 2, dropout=0.2
        super(TFEncoder, self).__init__()
        self.p = p
        self.feat_size, self.embed_size = feat_size, embed_size
        self.dropout, self.Tanh = nn.Dropout(p), nn.Tanh()
        self.encoder = nn.LSTM(76, self.embed_size, batch_first=True)
        self.pos_encoder = PositionalEncoding(embed_size, dropout=0.1)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, self.p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        print(f"n_heads:{nhead}, n_hid:{nhid}, n_layers:{num_layers}, dropout:{p}")

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, src_mask):
        # Convert data from (N, T, D) to (T, N, D)
        all_embeds, _ = self.encoder(x)
        all_embeds = all_embeds.transpose(0,1)
        #all_embeds = all_embeds * math.sqrt(self.embed_size)  
        
        # Add positional embeddings
        src = self.pos_encoder(all_embeds)   
         
        # Generate embeddings from Transformer Encoder 
        embeddings = self.transformer_encoder(src, src_mask)
        
        #Conver embeddings from (T, N, D) to (N, T, D)
        embeddings = embeddings.transpose(0, 1)
             
        return embeddings
  
#%% Category-wise LSTM Encoder

class CatewiseTemporalEncoder(nn.Module):  
    ''' Generate a temporal encoder for each channel'''
    def __init__(self, hidden_size, p=0.3, num_layers=2, dic_channels=dic_channels, device='cpu'):
        super(CatewiseTemporalEncoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p = p
        self.dropout = nn.Dropout(p)
        
        for k, c_list in dic_channels.items():
            setattr(self, 'encoder_'+str(k), CustomRNN(input_size=len(c_list), hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=self.p))
            print(k, c_list)
    def forward(self, x, lens):
        
        all_hts = []
        embs = []
        
        for k, v in dic_channels.items():
            encoder = getattr(self, 'encoder_'+str(k))
            x_k = torch.index_select(input=x, dim=-1, index=torch.tensor(v, device=self.device))
            cat_all_ht, (cat_embed, _) = encoder(x_k, lens)
            all_hts.append(cat_all_ht)
            embs.append(cat_embed[-1])
        
        cat_all_hts, cat_embs = torch.cat(all_hts, dim=-1), torch.cat(embs, dim=-1)
        cat_all_hts, cat_embs = self.dropout(cat_all_hts), self.dropout(cat_embs)
        
        return cat_all_hts, cat_embs

#%% Category-wise Encoder
''' Generate a temporal or transformer-based encoder for each category'''
class CatewiseEncoder(nn.Module):  
    def __init__(self, hidden_size, p=0.3, nlayers=2, dic_channels=dic_channels, device='cpu', nhead=4, dim_feedforward=512, 
                 encoder_type='LSTM'):
        super(CatewiseEncoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.p = p
        self.dropout = nn.Dropout(p)
        self.encoder_type = encoder_type
        
        # Create a list of encoders for each category
        for k, c_list in dic_channels.items():
            if self.encoder_type == 'LSTM':
                setattr(self, 'encoder_'+str(k), nn.LSTM(input_size=len(c_list), hidden_size=hidden_size, num_layers=nlayers, batch_first=True, dropout=p)) 
            elif self.encoder_type == 'GRU':
                setattr(self, 'encoder_'+str(k), nn.GRU(input_size=len(c_list), hidden_size=hidden_size, num_layers=nlayers, batch_first=True, dropout=p))
            
            elif self.encoder_type == 'Transformer':
                # Embedding layer   
                setattr(self, 'embedding_layer_'+str(k), nn.Linear(len(c_list), hidden_size))
                # Transformer encoder layer
                encoder_layer = TransformerEncoderLayer(hidden_size, nhead, dim_feedforward, dropout=p)
                # Transformer encoder
                setattr(self, 'encoder_'+str(k), TransformerEncoder(encoder_layer, nlayers))
            else:
                raise ValueError("Invalid encoder_type. Choose 'LSTM', 'GRU', or 'Transformer'.")
            #print(k, c_list)
        
        if self.encoder_type == 'Transformer':
            self.causal_mask = generate_causal_mask(48).to(self.device)
            print(self.causal_mask.shape)

    def forward(self, x, lens=None):
        all_hts, embs = [], []

        # Process each category in parallel
        for k, v in dic_channels.items():
            encoder = getattr(self, 'encoder_'+str(k))
            # Select the relevant features for the current category
            x_k = torch.index_select(input=x, dim=-1, index=torch.tensor(v, device=self.device))
            
            if self.encoder_type in ['LSTM', 'GRU']:
                cat_all_ht, (cat_embed, _) = encoder(x_k)
                all_hts.append(cat_all_ht)
                embs.append(cat_embed[-1])  # Get the last output for the category embedding

            elif self.encoder_type == 'Transformer':
                # Reshape x_k to (sequence_length, batch_size, feature_dim) for transformer
                x_k = x_k.transpose(0, 1)  # (batch_size, seq_length, feature_dim) -> (seq_length, batch_size, feature_dim)
                emb_k = getattr(self, 'embedding_layer_'+str(k))(x_k)  # Embedding
                cat_all_ht = encoder(emb_k,  mask=self.causal_mask)
                cat_all_ht = cat_all_ht.transpose(0, 1)
                all_hts.append(cat_all_ht)
                embs.append(cat_all_ht[-1])  # Get the last output for the category embedding

        # Concatenate results along the feature dimension
        cat_all_hts = torch.cat(all_hts, dim=-1)  # N x T x C*d_model
        cat_embs = cat_all_hts[:,-1,:]  # N x C*d_model
        cat_all_hts, cat_embs = self.dropout(cat_all_hts), self.dropout(cat_embs)

        # Stack the category embeddings
        stacked_all_hts = torch.stack(all_hts, dim=-2) # N x T x C x d_model
        stacked_embs = stacked_all_hts[:,-1,:,:] # N x C x d_model
        stacked_all_hts, stacked_embs = self.dropout(stacked_all_hts), self.dropout(stacked_embs)
        
        return cat_all_hts, cat_embs, stacked_all_hts, stacked_embs

        



#%% Self-Attention (Aborted)
'''
class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead=4, p=0.5, device='cpu'):
        super(SelfAttention, self).__init__()
        self.device = device

        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=p, batch_first=True, device=self.device)

    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool().to(self.device)
        return mask

    def forward(self, x, e):
        if mask:
            mask = self.generate_causal_mask(x.size(1))
        else:
            mask = None
        att_output, att_weights = self.attention(x, x, x, attn_mask=mask)
        return att_output, att_weights
'''
#%% Multi-Head Attention
class AttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, 
                 qk_hidden_dim, v_hidden_dim,
                 dropout=None):
        super(AttentionQKV, self).__init__()
        self.qk_hidden_dim = qk_hidden_dim
        self.v_hidden_dim = v_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(self.attention_input_dim, self.qk_hidden_dim)
        self.W_k = nn.Linear(self.attention_input_dim, self.qk_hidden_dim)
        self.W_v = nn.Linear(self.attention_input_dim, self.v_hidden_dim)

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))

        
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input):

        N, C, D = input.size() # Num_ts * Channels * Input_dim
        #print(f"input shape: {input.shape}")
        input_q = self.W_q(input) # N C H (H: Hidden_dim)
        input_k = self.W_k(input) # N C H
        input_v = self.W_v(input) # N C H

        q = torch.reshape(input_q, (N, self.qk_hidden_dim, C)) #N H C
        e = torch.matmul(input_k, q).squeeze()
        a = self.softmax(e/math.sqrt(self.qk_hidden_dim)) # Softmax(q*k^T/sqrt(H))
        v = torch.einsum('ncd, ndh->nch', a, input_v)
        v = self.dropout(v)

        return v, a

class MultiHeadAttention(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, nhead=4, p=0.5, device='cpu'):
        super(MultiHeadAttention, self).__init__()
        self.device = device
        self.attention_input_dim, self.attention_hidden_dim, self.nhead, self.p = attention_input_dim, attention_hidden_dim, nhead, p
        assert attention_input_dim % self.nhead == 0
        self.attention = clones(AttentionQKV(attention_input_dim=attention_input_dim, 
                                             qk_hidden_dim=self.attention_hidden_dim,
                                             v_hidden_dim=self.attention_hidden_dim//self.nhead,
                                             dropout=p).to(self.device), self.nhead)
        print(f'Multi-Head Attention with {self.nhead} heads')
    
    def forward(self, input):
        N, C, D = input.size() # Num_ts * Channels * Input_dim
        All_vs, All_as = [], []
        for att_head in self.attention:
            v_, a_ = att_head(input)
            All_vs.append(v_)
            All_as.append(a_)
        MHA_embed = torch.cat(All_vs, dim=-1)
        
        return MHA_embed, All_as


#%%
if __name__ == '__main__':
    data = torch.rand([10, 17, 32])
    model = AttentionQKV(attention_input_dim=32, attention_hidden_dim=32, dropout=0.5)
    v, a = model(data)
    print(v.shape, a[0,0,:].sum(), a[0,0,:])

    model2 = MultiHeadAttention(attention_input_dim=32, attention_hidden_dim=32, nhead=1, p=0.5, device='cpu')
    MHA_embed, All_as = model2(data)
    print(MHA_embed.shape, len(All_as))


# %%
class EHREncoder(nn.Module):
    def __init__(self, d_model, encoder_type='LSTM',  p=0.5, nlayers=2, nhead=4, d_ff=256, device='cpu', 
                 skip_connection=True, PFF=False):
        super(EHREncoder, self).__init__()
        self.encoder_type = encoder_type
        self.d_model, self.nhead, self.p, self.d_ff = d_model, nhead, p, d_ff
        self.dim_ff_input = d_model*len(dic_channels)
        self.skip_connection = skip_connection
        self.PFF = PFF
        # Components
        self.encoder = CatewiseEncoder(hidden_size=self.d_model, encoder_type=self.encoder_type, p=self.p, nlayers=nlayers, device=device)
        self.MHAtt = MultiHeadAttention(attention_input_dim=self.d_model, attention_hidden_dim=self.d_model, nhead=self.nhead, p=self.p, device=device)
        if self.PFF:    
            self.PositionwiseFeedForward = PositionwiseFeedForward(d_model=self.dim_ff_input, d_ff=self.d_ff, dropout=self.p)
        self.output = nn.Linear(self.dim_ff_input, self.dim_ff_input, device=device)

        self.Dropout = nn.Dropout(self.p)
        self.Relu=nn.ReLU()

        if self.skip_connection:
            print('Skip connection is enabled')
        if self.PFF:
            print(f'Position-wise feedforward is enabled with hidden dimension {self.d_ff}')
    
    def forward(self, x):
        # Category-wise Temporal Encoder  
        _, _, stacked_all_hts, _ = self.encoder(x)
        
        # Multi-Head Attention
        N, T, C, D = stacked_all_hts.shape
        trans_all_hts = stacked_all_hts.reshape((N*T, C, D))
        MHA_all_hts, All_as = self.MHAtt(trans_all_hts)
        ## Skip connection
        if self.skip_connection:
            MHA_all_hts = trans_all_hts + MHA_all_hts # N*T C D
        MHA_all_hts = MHA_all_hts.reshape((N, T, C*D)) # N T C*D

        # PositionwiseFeedForward + Linear projection
        if self.PFF:
            ff_output = self.PositionwiseFeedForward(MHA_all_hts)[0]
            if self.skip_connection:
                ff_output = MHA_all_hts + self.Dropout(ff_output)
            linear_output = self.Dropout(self.Relu(self.output(ff_output)))
            if self.skip_connection:
                linear_output = ff_output + linear_output
        else:
            linear_output = MHA_all_hts

        all_hts, final_ht = linear_output, linear_output[:,-1,:]
        return all_hts, final_ht


class EHREncoder2(nn.Module):
    def __init__(self, input_size, hidden_size, attention_input_dim, attention_hidden_dim, nhead, nlayers,
                 bidirectional, p):
        super(EHREncoder2, self).__init__()
        self.attention_hidden_dim = attention_hidden_dim
        # LSTM层定义，支持单向和双向
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, num_layers=nlayers, batch_first=True, dropout=p)

        # Multi-Head Attention层
        self.multihead_attention = MultiHeadAttention(attention_input_dim=attention_input_dim,
                                                      attention_hidden_dim=attention_hidden_dim,
                                                      nhead=nhead,
                                                      p=p)

    def forward(self, x):
        # 输入x的形状是 (N, T, D) -> (batch_size, seq_length, input_dim)

        # LSTM输出
        lstm_out, (h_n, c_n) = self.lstm(x)  # (N, T, lstm_hidden_dim)

        # 双向LSTM需要拼接最后两个时间步的输出
        if self.lstm.bidirectional:
            lstm_out = torch.cat((lstm_out[:, :, :self.lstm.hidden_size], lstm_out[:, :, self.lstm.hidden_size:]),
                                 dim=-1)

        # 然后输入到MultiHead Attention层
        mha_out, _ = self.multihead_attention(lstm_out)  # (N, T, attention_dim)

        return mha_out

class EHREncoder3(nn.Module):
    def __init__(self, d_model, encoder_type='LSTM', p=0.5, nlayers=2, nhead=4, d_ff=256, device='cpu',
                 skip_connection=True, PFF=False):
        super(EHREncoder3, self).__init__()
        self.encoder_type = encoder_type
        self.d_model, self.nhead, self.p, self.d_ff = d_model, nhead, p, d_ff
        self.dim_ff_input = d_model * len(dic_channels)  # C * H
        self.skip_connection = skip_connection
        self.PFF = PFF
        self.device = device

        # 类别编码器
        self.encoder = CatewiseEncoder(hidden_size=self.d_model, encoder_type=self.encoder_type,
                                       p=self.p, nlayers=nlayers, device=device)

        # 变量注意力（在 C 维度）
        self.var_attention = MultiHeadAttention(attention_input_dim=self.d_model,
                                                attention_hidden_dim=self.d_model,
                                                nhead=self.nhead,
                                                p=self.p,
                                                device=device)

        # 时间注意力（在 T 维度）
        self.time_attention = MultiHeadAttention(attention_input_dim=self.d_model,
                                                 attention_hidden_dim=self.d_model,
                                                 nhead=self.nhead,
                                                 p=self.p,
                                                 device=device)

        # 可选前馈网络
        if self.PFF:
            self.PositionwiseFeedForward = PositionwiseFeedForward(d_model=self.dim_ff_input,
                                                                    d_ff=self.d_ff,
                                                                    dropout=self.p)

        self.output = nn.Linear(self.dim_ff_input, self.dim_ff_input, device=device)
        self.Dropout = nn.Dropout(self.p)
        self.Relu = nn.ReLU()

        if self.skip_connection:
            print('✅ Skip connection is enabled')
        if self.PFF:
            print(f'✅ Position-wise FeedForward enabled with hidden dim {self.d_ff}')
        print('✅ Variable Attention & Temporal Attention enabled')

    def forward(self, x):
        
        _, _, stacked_all_hts, _ = self.encoder(x)  # (N, T, C, H)

        N, T, C, H = stacked_all_hts.shape

        # 变量注意力
        var_in = stacked_all_hts.reshape(N * T, C, H)  # (N*T, C, H)
        var_att_out, _ = self.var_attention(var_in)    # (N*T, C, H)
        if self.skip_connection:
            var_att_out = var_in + var_att_out
        var_att_out = var_att_out.reshape(N, T, C, H)  # (N, T, C, H)

        # 时间注意力
        time_in = var_att_out.permute(0, 2, 1, 3)         # (N, C, T, H)
        time_in = time_in.reshape(N * C, T, H)            # (N*C, T, H)
        time_att_out, _ = self.time_attention(time_in)    # (N*C, T, H)
        if self.skip_connection:
            time_att_out = time_in + time_att_out
        time_att_out = time_att_out.reshape(N, C, T, H).permute(0, 2, 1, 3)  # (N, T, C, H)


        concat_hts = time_att_out.reshape(N, T, C * H)  # (N, T, C*H)


        if self.PFF:
            ff_output = self.PositionwiseFeedForward(concat_hts)[0]
            if self.skip_connection:
                ff_output = concat_hts + self.Dropout(ff_output)
            linear_output = self.Dropout(self.Relu(self.output(ff_output)))
            if self.skip_connection:
                linear_output = ff_output + linear_output
        else:
            linear_output = concat_hts  # (N, T, C*H)

        all_hts = linear_output
        final_ht = linear_output[:, -1, :]  # 最后时间步的输出

        return all_hts, final_ht



# %%
if __name__ == '__main__':
    data = torch.rand([10, 48, 76])
    Encoder = EHREncoder(8, encoder_type='LSTM', device='cpu')
    all_hts, final_ht = Encoder(data)
    print(all_hts.shape, final_ht.shape)

# %%
if __name__ == '__main__':
    data = torch.rand([10, 48, 76])
    Encoder = CatewiseEncoder(8, encoder_type='LSTM', device='cpu')
    cat_all_hts, cat_embed, stack_all_hts, stack_embed = Encoder(data)
    print(cat_all_hts.shape, cat_embed.shape, stack_all_hts.shape, stack_embed.shape)

# %% Check shape conversion
if __name__ == '__main__':
    N, T, C, D = 20, 5, 16, 32
    ori_data = torch.rand([N, T, C, D])
    trans_data1 = ori_data.reshape((N*T, C, D))
    trans_data2 = ori_data.reshape((N, T, C*D))

    assert trans_data1[2*T,1,0] == trans_data2[2,0,D]
    

# %%



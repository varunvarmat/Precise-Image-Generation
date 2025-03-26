import torch
import torch.nn as nn
import torch.nn.functional as F
class BlockLayer(nn.Module):
    def __init__(self, attn, norm1, feed_forward,  norm2, up_sample_linear, dropout=0.1):
        super(BlockLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.norm1, self.norm2 = norm1, norm2
        self.re_sample_linear=up_sample_linear

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # multihead attn & norm
        a = self.attn(x)
        t = self.norm1(x + self.dropout1(a))

        # feed forward & norm
        z = self.feed_forward(t)  # linear(dropout(act(linear(x)))))
        y = self.norm2(t + self.dropout2(z))

        f= self.re_sample_linear(y)

        return f
    
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size, dropout):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size) #(input_size,head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x) #5*64
        K = self.key(x)
        V = self.value(x)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2,-1) #5*5
        print(attention.shape)

        # Scaling
        attention = attention / (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1) #5*5
        attention = self.dropout(attention)
        # attention_maps = attention[:,0]
        # #print(attention_maps.size())
        # attention_maps = attention_maps[:,1:]
        # n_samples,first_row=attention_maps.size()
        # #print(attention_maps.size())
        # #first_row=attention_maps[0]
        # side_length = int(first_row ** 0.5)
        # attention_maps = attention_maps.view(n_samples,side_length, side_length)
        #print(attention_maps.size())


        out = attention @ V #5*64

        #return out,attention_maps
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.dropout=dropout
        assert d_model % n_heads == 0, "d_model is not divisible by the heads"
        self.head_size = d_model // n_heads
        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size, self.dropout) for _ in range(n_heads)])
        self.linear = nn.Linear(d_model, d_model)
        self.n_heads=n_heads

    def forward(self, x):
        # Combine attention heads
        head_outputs=[]
        #attention_maps=None
        for head in self.heads:
            # out,attention=head(x)
            # head_outputs.append(out)
            # if attention_maps is None:
            #     attention_maps=attention
            # else:
            #     attention_maps=attention_maps+attention
            out=head(x)
            head_outputs.append(out)
        #attention_maps=attention_maps/self.n_heads
        
        #attention_maps=torch.stack(attention_maps) #4*32*5*5
        output = torch.cat(head_outputs, dim=-1) #5*64->5*256
        output = self.linear(output)
        #return output,attention_maps
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dr_ff, dropout=0.1, act='relu', d_output=None):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = dr_ff*d_model
        d_output = d_model if d_output is None else d_output

        self.ffn_1 = nn.Linear(d_model, self.d_ff)
        self.ffn_2 = nn.Linear(self.d_ff, d_output)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ffn_2(self.dropout(self.act(self.ffn_1(x))))
        return y

class Encoder(nn.Module):
    def __init__(self, d_model ,dr_ff, N, no_head, schedule, dropout=0.1):
        super(Encoder, self).__init__()
        self.N = N #number of encoder blocks
        self.layers = nn.ModuleList()
        self.d_model=d_model 
        self.upsample_schedule=schedule #eg:[32,64]
        #print(self.upsample_schedule)
        assert self.N==len(self.upsample_schedule), "schedule and number of layers should be same"
        for i in range(N):
            #print(self.d_model)
            #print(self.upsample_schedule[i])
            self.layers.append(
                BlockLayer(MultiHeadAttention(self.d_model, no_head, dropout=dropout),
                        nn.LayerNorm(self.d_model),
                        FeedForward(self.d_model, dr_ff, dropout=dropout),
                        nn.LayerNorm(self.d_model),
                        nn.Linear(self.d_model, self.upsample_schedule[i])
                        ))
            self.d_model=self.upsample_schedule[i]
            
            


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
class Decoder(nn.Module): #autoencoder not vae, need to l2 regularize the latent.
    def __init__(self, d_model ,d_ff, N, no_head, schedule, dropout=0.1):
        super(Decoder, self).__init__()
        self.N = N #number of encoder blocks
        self.layers = nn.ModuleList()
        self.d_model=d_model #eg 128
        #print(self.d_model)
        self.downsample_schedule=schedule #eg:[32,16]
        #print(self.downsample_schedule)
        
        assert self.N==len(self.downsample_schedule), "schedule and number of layers should be same"
        for i in range(N):
            self.layers.append(
                BlockLayer(MultiHeadAttention(self.d_model, no_head, dropout=dropout),
                        nn.LayerNorm(self.d_model),
                        FeedForward(self.d_model, d_ff, dropout=dropout),
                        nn.LayerNorm(self.d_model),
                        nn.Linear(self.d_model, self.downsample_schedule[i])
                        ))
            self.d_model=self.downsample_schedule[i]
            


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, d_model ,dr_ff, N, no_head, schedule, dropout=0.1):
        super(AutoEncoder, self).__init__()
        self.up_sample_schedule=schedule[1:]
        #print(self.up_sample_schedule)
        self.down_sample_schedule=schedule[::-1][1:]
        #print(self.down_sample_schedule)
        # self.d_model=d_model
        # self.latent_size=schedule[-1]

        self.encoder=Encoder(d_model,dr_ff,N,no_head,self.up_sample_schedule,dropout)
        d_model=schedule[-1]
        self.decoder=Decoder(d_model,dr_ff,N,no_head,self.down_sample_schedule,dropout)
    
    def forward(self,x):
        latent=self.encoder(x)
        #how to l2 regularize the latent
        x_hat=self.decoder(latent)
        return x_hat, latent
        # return x_hat

    

    


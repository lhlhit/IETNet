import math
import torch
import torch.nn as nn
from models.ietnet_utils import PointNetSetAbstraction
import matplotlib.pyplot as plt
import seaborn as sns

def create_model_ietnet(cfgs):
    """ 构建模型 """
    model = IETNet(
        loc_size=cfgs['IETNet']['loc_size'],
        dim_model=cfgs['IETNet']['dim_model'],
        n_heads=cfgs['IETNet']['n_heads'],
        n_layers=cfgs['IETNet']['n_layers'],
        dropout=cfgs['IETNet']['dropout'],
        scale=cfgs['IETNet']['scale'],
        TIME = cfgs['IETNet']['Time_Length']
    )
    return model


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        self.hidden_size = d_model
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(1000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.hidden_size)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # 线性变换层
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 注意力输出层
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, seq_inputs, key, value):
        # 执行线性变换以获取查询、键和值
        Q = self.query_linear(seq_inputs)
        K = self.key_linear(seq_inputs)
        V = self.value_linear(seq_inputs)

        # 将查询、键和值分割为多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        # for i in range(self.num_heads):
        #     a = attention_weights.cpu().detach().numpy()[0][i]
        #     plt.imshow(a, cmap="jet")
        #     plt.colorbar()
        #     plt.show()
        # 对注意力权重加权求和
        attention_output = torch.matmul(attention_weights, V)
        # 将多个头的输出连接起来
        attention_output = self.combine_heads(attention_output)
        # 执行输出线性变换
        attention_output = self.output_linear(attention_output)
        return attention_output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, self.num_heads * d_k)
        return x


    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, self.num_heads * d_k)
        return x


class FeedForward(nn.Module):

    def __init__(self, dim_in, dim_out, dropout=0.1):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_out, dim_out),
            nn.LayerNorm(dim_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_out, dim_in)
        )
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, x):
        outs = self.feed_forward(x)
        outs = self.norm(outs + x)
        return outs


class Encoder(nn.Module):

    def __init__(self, dim_model, n_heads=4, dropout=0.1, scale=1):
        super(Encoder, self).__init__()
        self.attn = MultiHeadAttention(dim_model, n_heads)
        self.feed_forward = FeedForward(dim_model, dim_model // scale, dropout)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, query=None, mask=None):
        attn = self.attn(x, query, mask)
        outs = self.feed_forward(attn)
        outs = self.norm(outs)
        return outs


class ImpEncoder(nn.Module):

    def __init__(self, dim_model, n_heads=4, n_layers=8, dropout=0.1, scale=1):
        super(ImpEncoder, self).__init__()
        self.encoders = nn.ModuleList([
            Encoder(dim_model, n_heads, dropout, scale) for _ in range(n_layers)
        ])

    def forward(self, x, query=None, mask=None) -> torch.Tensor:
        for layer in self.encoders:
            x = layer(x, query, mask)
        return x


class Decoder(nn.Module):

    def __init__(self, loc_size, encode_dim, hidden_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encode_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, loc_size),
        )

    def forward(self, encoder_x):
        return self.mlp(encoder_x)


class IETNet(nn.Module):

    def __init__(self, loc_size, dim_model,
                 n_heads, n_layers, TIME, dropout=0.1, scale=1):
        super(IETNet, self).__init__()
        in_channel = 4
        self.normal_channel = True
        self.sa1 = PointNetSetAbstraction(TIME, npoint=128, radius=4, nsample=8, in_channel=in_channel, mlp=[64, 128],
                                          group_all=False)
        self.sa0 = PointNetSetAbstraction(TIME, npoint=None, radius=None, nsample=None, in_channel=in_channel,
                                          mlp=[256, 512, 1024], group_all=True)

        self.fc = nn.Sequential(nn.Linear(128, dim_model))

        self.token_embed = nn.Linear(int(loc_size), dim_model)
        nn.init.xavier_uniform_(self.token_embed.weight)
        nn.init.zeros_(self.token_embed.bias)
        self.token_norm = nn.LayerNorm(dim_model)
        self.pos_embed = PositionalEmbedding(dim_model, TIME)
        self.encoder = ImpEncoder(dim_model, n_heads, n_layers,
                                   dropout, scale)
        self.decoder = Decoder(loc_size, dim_model, dim_model, dropout)

    def forward(self, xyz, mask=None):
        # 3D-SIP
        if self.normal_channel:
            norm = xyz[:,:,:, 3:]
            xyz0 = xyz[:,:,:, :3]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz0, norm)
        l3_points = l1_points.permute(0, 2, 1)
        l3_points = self.fc(l3_points)

        # ImpFormer
        tokens_embed = self.token_norm(l3_points)
        pos_embed = self.pos_embed(tokens_embed)
        embeded = pos_embed

        # encoder
        encoder_x = self.encoder(embeded)

        # decoder
        decoder_x = self.decoder(encoder_x)

        return decoder_x

if __name__ == '__main__':
    model = IETNet(20, 384, 6, 2, 2).to('cuda')
    xyz = torch.rand(3,2, 20, 4).to('cuda')  # batchsize, time, pointnum, feature
    # xyz = xyz.permute(1, 0, 3, 2)

    out = (model(xyz))

    print('end')


from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

class MotionChunkVAE_Transformer(nn.Module):
    def __init__(
            self,
            num_heads,
            num_layers,
            enc_dim,
            input_dim,
            dim_ff,
            context_len,
            dropout,
            *args, 
            **kwargs
            ):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.enc_dim = enc_dim
        self.dim_ff = dim_ff
        self.context_len = context_len
        self.input_dim = input_dim
        
        self.frame_encoder = nn.Sequential(
          nn.Linear(self.input_dim, self.enc_dim),
          nn.LayerNorm(self.enc_dim),
          nn.SELU(),
          nn.Dropout(dropout),
          nn.Linear(self.enc_dim, self.enc_dim),
          nn.LayerNorm(self.enc_dim),
          nn.SELU(),
          nn.Dropout(dropout),
          nn.Linear(self.enc_dim, self.enc_dim),
          nn.LayerNorm(self.enc_dim),
          nn.SELU(),
          nn.Dropout(dropout),
        )
        
        self.input_pos_embedding = nn.Embedding(self.context_len, self.enc_dim)
        self.sequence_encoder_layer = nn.TransformerEncoderLayer(d_model=self.enc_dim, nhead=self.num_heads, dim_feedforward=self.dim_ff, dropout=dropout, batch_first=True)
        self.sequence_encoder = nn.TransformerEncoder(self.sequence_encoder_layer, num_layers=self.num_layers)

        self.output_pos_embedding = nn.Embedding(self.context_len, self.enc_dim)
        self.sequence_decoder_layer = nn.TransformerEncoderLayer(d_model=self.enc_dim, nhead=self.num_heads, dim_feedforward=self.dim_ff, dropout=dropout, batch_first=True)
        self.sequence_decoder = nn.TransformerEncoder(self.sequence_decoder_layer, num_layers=self.num_layers)

        self.frame_decoder = nn.Sequential(
          nn.Linear(self.enc_dim, self.enc_dim),
          nn.LayerNorm(self.enc_dim),
          nn.SELU(),
          nn.Dropout(dropout),
          nn.Linear(self.enc_dim, self.input_dim),
        )
        
    def encode(self, motion_chunk, motion_mask):
        res = self.frame_encoder(motion_chunk)
        res = torch.sum(res, dim=1)
        return res
        res = res + self.input_pos_embedding(torch.arange(self.context_len, device=motion_chunk.device))[None, ...].repeat(motion_chunk.shape[0], 1, 1)
        res = self.sequence_encoder(src = res)
        res = torch.sum(res * motion_mask[..., None], dim=1)
        return res
    
    def decode(self, enc):
        res = enc[:, None, :].repeat(1, self.context_len, 1) + self.output_pos_embedding(torch.arange(self.context_len, device=enc.device))[None, ...].repeat(enc.shape[0], 1, 1)
        res = self.sequence_decoder(res)
        res = self.frame_decoder(res)
        return res
        
    def forward(self, motion_chunk, motion_mask):
        enc = self.encode(motion_chunk, motion_mask)
        dec = self.decode(enc)
        return enc, dec

'''
class MotionChunkVAE_Transfomer_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.num_layers = 3
        self.enc_dim = 256
        self.dim_ff = 1024
        self.context_len = 16
        self.frame_encoder = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(263,self.enc_dim)),
          ('relu1', nn.ReLU()),
          ('dropout1', nn.Dropout(0.1)),
          ('linear2', nn.Linear(self.enc_dim,self.enc_dim)),
          ('relu2', nn.ReLU()),
          ('dropout2', nn.Dropout(0.1)),
        ]))
        
        self.input_pos_embedding = nn.Embedding(self.context_len, self.enc_dim)
        self.input_pos_inds = nn.parameter.Parameter(torch.arange(0, self.context_len), requires_grad=False)
        self.sequence_encoder_layer = nn.TransformerEncoderLayer(d_model=self.enc_dim, nhead=self.num_heads, dim_feedforward = self.dim_ff, batch_first = True)
        self.sequence_encoder = nn.TransformerEncoder(self.sequence_encoder_layer, num_layers = self.num_layers)

        self.output_pos_embedding = nn.Embedding(self.context_len, self.enc_dim)
        self.sequence_decoder_layer = nn.TransformerEncoderLayer(d_model=self.enc_dim, nhead=self.num_heads, dim_feedforward = self.dim_ff, batch_first = True)
        self.sequence_decoder = nn.TransformerEncoder(self.sequence_decoder_layer, num_layers=self.num_layers)

        self.frame_decoder = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(self.enc_dim,self.enc_dim)),
          ('relu1', nn.ReLU()),
          ('dropout1', nn.Dropout(0.1)),
          ('linear2', nn.Linear(self.enc_dim,263)),
        ]))

        # distribution parameters
        self.fc_mu = nn.Linear(self.enc_dim, self.enc_dim)
        self.fc_var = nn.Linear(self.enc_dim, self.enc_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, recon, logscale, input):
        scale = torch.exp(logscale)
        mean = recon
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(input)
        return log_pxz.sum(dim=(1, 2))
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
        
    def encode(self, motion_chunk, motion_mask):
        square_motion_mask = motion_mask[..., None].repeat(self.num_heads, 1, motion_mask.shape[1])
        square_motion_mask = torch.where(torch.logical_and(square_motion_mask, torch.transpose(square_motion_mask, 2, 1)), 1.0, 0.0)
        res = self.frame_encoder(motion_chunk)
        res = res + self.input_pos_embedding(self.input_pos_inds)[None, ...].repeat(motion_chunk.shape[0], 1, 1)
        res = self.sequence_encoder(res, square_motion_mask)
        res = torch.sum(res * motion_mask[..., None], dim=1)
        return res
    
    def decode(self, enc):
        res = enc[:, None, :].repeat(1, self.context_len, 1) + self.output_pos_embedding(self.input_pos_inds)[None, ...].repeat(enc.shape[0], 1, 1)
        res = self.sequence_decoder(res)
        res = self.frame_decoder(res)
        return res
        
    def forward(self, motion_chunk, motion_mask):
        enc = self.encode(motion_chunk, motion_mask)
        dec = self.decode(enc)
        return enc, dec
    
'''

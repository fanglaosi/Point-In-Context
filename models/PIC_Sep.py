import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils.logger import *
from pytorch3d.ops import sample_farthest_points, knn_points

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            idx: B G M
        '''
        # fps the centers out
        center, center_idx = sample_farthest_points(xyz, K=self.num_group) # [B, npoint, 3]  [B, npoint]

        # knn to get the neighborhood
        _, neighborhood_idx, _ = knn_points(center, xyz, K=self.group_size, return_nn=False) # [B, npoint, k]

        assert neighborhood_idx.size(1) == self.num_group
        assert neighborhood_idx.size(2) == self.group_size

        return center_idx, neighborhood_idx

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x):
        merge_idx = 2
        for idx, block in enumerate(self.blocks):
            if idx == merge_idx:
                x = (x[:x.shape[0] // 2] + x[x.shape[0] // 2:]) * 0.5
            x = block(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x)

        x = self.head(self.norm(x))  # only return the mask tokens predict pixel
        return x

class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_rand(self, center):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape


        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, pc_center, pc_neighborhood, target_center, target_neighborhood, pos):
        # generate mask
        if self.training:
            bool_masked_pos = self._mask_center_rand(target_center) # B 2G
        else:
            vis = torch.zeros((pc_center.shape[0], pc_center.shape[1]//2))
            mask = torch.ones((pc_center.shape[0], pc_center.shape[1]//2))
            bool_masked_pos = torch.cat([vis, mask], dim=1).to(torch.bool) # B 2G
        group_pc_tokens = self.encoder(pc_neighborhood)  # B 2G C
        group_target_tokens = self.encoder(target_neighborhood)  # B 2G C

        batch_size, seq_len, C = group_target_tokens.size()
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)  # B 2G C
        m = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(batch_size, seq_len, 1)  # B 2G 1
        group_target_tokens = group_target_tokens * (1 - m) + mask_token * m  # B 2G C

        x = torch.cat((group_pc_tokens, group_target_tokens), dim=0)  # 2B 2G C

        pos = pos.to(x.device)
        x = x + pos

        # transformer
        x = self.blocks(x)  # B 2G C
        x = self.norm(x)

        return x, bool_masked_pos

@MODELS.register_module()
class PIC_Sep(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PIC-Sep] ', logger ='PIC-Sep')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[PIC-Sep] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PIC-Sep')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)

        self.pos_sincos = self.get_positional_encoding(2 * self.num_group, self.trans_dim)

        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def combine_center(self, pc1, pc2):
        return torch.cat([pc1, pc2], dim=1)

    def combine_neighborhood(self, pc1, pc2):
        return torch.cat([pc1, pc2], dim=1)

    def get_positional_encoding(self, max_seq_len, embed_dim):
        positional_encoding = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(embed_dim):
                if i % 2 == 0:
                    positional_encoding[pos, i] = torch.sin(pos / torch.tensor(10000 ** (2 * i / embed_dim)))
                else:
                    positional_encoding[pos, i] = torch.cos(pos / torch.tensor(10000 ** (2 * i / embed_dim)))
        return positional_encoding

    def joint_sampling(self, pc1, pc2, target1, target2):
        # Farthest Points Sample
        pc1_center, pc1_center_idx = sample_farthest_points(pc1, K=self.num_group)
        pc2_center, pc2_center_idx = sample_farthest_points(pc2, K=self.num_group)

        _, pc1_neighborhood_idx, _ = knn_points(pc1_center, pc1, K=self.group_size, return_nn=False)
        _, pc2_neighborhood_idx, _ = knn_points(pc2_center, pc2, K=self.group_size, return_nn=False)
        pc1_neighborhood = index_points(pc1, pc1_neighborhood_idx)
        pc2_neighborhood = index_points(pc2, pc2_neighborhood_idx)

        target1_center = index_points(target1, pc1_center_idx)
        target2_center = index_points(target2, pc2_center_idx)
        _, target1_neighborhood_idx, _ = knn_points(target1_center, target1, K=self.group_size, return_nn=False)
        _, target2_neighborhood_idx, _ = knn_points(target2_center, target2, K=self.group_size, return_nn=False)
        target1_neighborhood = index_points(target1, target1_neighborhood_idx)
        target2_neighborhood = index_points(target2, target2_neighborhood_idx)

        return pc1_center, pc1_neighborhood, pc2_center, pc2_neighborhood, target1_center, target1_neighborhood, target2_center, target2_neighborhood

    def forward(self, pc1, pc2, target1, target2, **kwargs):
        pc1_center, pc1_neighborhood, pc2_center, pc2_neighborhood, target1_center, \
        target1_neighborhood, target2_center, target2_neighborhood = self.joint_sampling(pc1, pc2, target1, target2)

        pc_center = self.combine_center(pc1_center, pc2_center) # [B 2G 3]
        pc_neighborhood = self.combine_neighborhood(pc1_neighborhood, pc2_neighborhood)

        target_center = self.combine_center(target1_center, target2_center) # [B 2G 3]
        target_neighborhood = self.combine_neighborhood(target1_neighborhood, target2_neighborhood)

        x, mask = self.MAE_encoder(pc_center, pc_neighborhood, target_center, target_neighborhood, self.pos_sincos) # x[B 2G C] mask[B 2G]

        B, seq_len, C = x.shape # B 2G C

        pos = self.pos_sincos.to(x.device)

        x = x + pos
        x = self.MAE_decoder(x) # B 2G C

        x_rec = x[mask].reshape(B, -1, C)
        B, M, C = x_rec.shape
        rebuild_target = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B*M Groupsize 3
        gt_points = target_neighborhood[mask].reshape(B*M, -1, 3)  # B*M Groupsize 3

        loss = self.loss_func(rebuild_target, gt_points)

        vis_teaget = target_neighborhood[~mask].reshape(B * (2 * self.num_group - M), -1, 3) # B*(2G-M) groupsize 3
        full_target = torch.cat([vis_teaget, rebuild_target], dim=0) # B*2G groupsize 3
        full_target = full_target.reshape(B, 2*self.num_group*self.group_size, 3)  # B 2*G*groupsize 3
        rebuild_target = rebuild_target.reshape(B, M * self.group_size, 3) # B M*groupsize 3
        gt_points = gt_points.reshape(B, M * self.group_size, 3) # B M*groupsize 3

        # for visualization
        # target2_xyz = pc2_neighborhood.reshape(B, M * self.group_size, 3) # B M*groupsize 3

        return gt_points, rebuild_target, loss

        # for visualization
        # return full_target, rebuild_target, gt_points, target2_xyz
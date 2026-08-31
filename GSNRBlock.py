# ===================== GSNR=====================
class GSNRBlock(nn.Module):
    def __init__(self, embed_dim:int=256, num_heads:int=8, ffn_ratio:int=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.offset_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim//2, 2)
        )

        self.fuse_proj = nn.Linear(embed_dim*2, embed_dim)

    def bilinear_sample_feat(self, feat_map:torch.Tensor, points_norm:torch.Tensor):
        B,N,_ = points_norm.shape
        grid = points_norm.view(B, N, 1, 2)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=False)
        sampled = sampled.squeeze(-1)
        sampled = sampled.permute(0,2,1)
        return sampled

    def forward(self, coords:torch.Tensor, embeds:torch.Tensor, Fsd:torch.Tensor):
        B,N,C = embeds.shape
        img_feat = self.bilinear_sample_feat(Fsd, coords)
        fuse_feat = torch.cat([embeds, img_feat], dim=-1)
        x = self.fuse_proj(fuse_feat)

        x_t = x.transpose(0,1)
        attn_out, _ = self.self_attn(x_t, x_t, x_t)
        x = x_t + attn_out
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        x = x.transpose(0,1)
        new_embeds = x

        delta_coords = self.offset_mlp(new_embeds)
        new_coords = coords + delta_coords
        new_coords = torch.clamp(new_coords, min=-1.0, max=1.0)
        return new_coords, new_embeds

class GSNR(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_iter:int=3, ffn_ratio=4):
        super().__init__()
        self.num_iter = num_iter
        self.blocks = nn.ModuleList([
            GSNRBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio)
            for _ in range(num_iter)
        ])

    def forward(self, init_coords:torch.Tensor, init_embeds:torch.Tensor, Fsd:torch.Tensor):
        coords = init_coords
        embeds = init_embeds
        for blk in self.blocks:
            coords, embeds = blk(coords, embeds, Fsd)
        return coords, embeds

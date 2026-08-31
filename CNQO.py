# ===================== CNQO =====================
class CNQO(nn.Module):
    def __init__(self, embed_dim=256, grid_scale:int=32, Nmax:int=200):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_scale = grid_scale
        self.Nmax = Nmax

        self.conv1_proj = conv1x1(embed_dim, embed_dim//2)
        self.ups_2x = UpsampleBlock(scale_factor=2)
        self.conv3_out = conv3x3(embed_dim//2, 2)
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def generate_uniform_grid(self, B:int, device):
        S = self.grid_scale
        ys, xs = torch.meshgrid(torch.linspace(-1,1,S,device=device),
                                torch.linspace(-1,1,S,device=device),
                                indexing="ij")
        grid_points = torch.stack([xs, ys], dim=-1)
        grid_points = grid_points.reshape(-1,2)
        grid_points = grid_points.unsqueeze(0).repeat(B,1,1)
        return grid_points, None

    def bilinear_sample_feat(self, feat_map:torch.Tensor, points_norm:torch.Tensor):
        B,N,_ = points_norm.shape
        grid = points_norm.view(B, N, 1, 2)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=False)
        sampled = sampled.squeeze(-1)
        sampled = sampled.permute(0,2,1)
        return sampled

    def forward(self, Fsd:torch.Tensor):
        B,C,H,W = Fsd.shape
        device = Fsd.device
        grid_coords, _ = self.generate_uniform_grid(B, device)
        init_query_embeds = self.bilinear_sample_feat(Fsd, grid_coords)

        x = self.conv1_proj(Fsd)
        x = self.ups_2x(x)
        x = self.conv3_out(x)
        x = F.softmax(x, dim=1)
        Mfore = x[:,0:1,...]
        Mdens = x[:,1:2,...]

        Sfd_map = self.alpha * Mfore + (1.0 - self.alpha) * Mdens
        sfd_per_point = self.bilinear_sample_feat(Sfd_map, grid_coords)
        score = sfd_per_point.squeeze(-1)

        _, top_idx = torch.topk(score, k=self.Nmax, dim=-1)
        query_coords = torch.gather(grid_coords, dim=1, index=top_idx[...,None].expand(-1,-1,2))
        query_embeds = torch.gather(init_query_embeds, dim=1, index=top_idx[...,None].expand(-1,-1,self.embed_dim))
        return query_coords, query_embeds, Mfore, Mdens, Sfd_map
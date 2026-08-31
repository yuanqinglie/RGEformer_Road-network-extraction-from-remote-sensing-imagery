# ===================== MSTR=====================
class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer: node self‑attention + FFN"""
    def __init__(self, embed_dim:int, num_heads:int, ffn_ratio:int=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim)
        )

    def forward(self, node_feat:torch.Tensor):
        """node_feat: [B,N,C]"""
        B,N,C = node_feat.shape
        x_t = node_feat.transpose(0,1) # [N,B,C]
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x = x_t + attn_out
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        out = x.transpose(0,1) # [B,N,C]
        return out


class CoordPosEncoder(nn.Module):
    """Project 2D coordinates into positional encoding and add to node features"""
    def __init__(self, embed_dim:int):
        super().__init__()
        self.proj = nn.Linear(2, embed_dim)

    def forward(self, coords:torch.Tensor):
        """coords: [B,N,2]"""
        pos_emb = self.proj(coords) # [B,N,C]
        return pos_emb


class EdgePredictor(nn.Module):
    """Edge prediction head: takes features of node i,j and outputs edge existence logit"""
    def __init__(self, embed_dim:int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, feat_i:torch.Tensor, feat_j:torch.Tensor):
        concat = torch.cat([feat_i, feat_j], dim=-1)
        logit = self.mlp(concat)
        return logit.squeeze(-1)


class MSTR(nn.Module):
    def __init__(
        self,
        embed_dim:int=256,
        num_heads:int=8,
        num_layers:int=3,
        ffn_ratio:int=4
    ):
        super().__init__()
        self.pos_encoder = CoordPosEncoder(embed_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, ffn_ratio)
            for _ in range(num_layers)
        ])
        self.edge_predictor = EdgePredictor(embed_dim)

    def forward(self, node_coords:torch.Tensor, node_embeds:torch.Tensor):
        """
        node_coords: [B,N,2] normalized coordinates in range [-1,1]
        node_embeds: [B,N,C]
        return edge_logits: [B,N,N] adjacency matrix logits, diagonal filled with ‑inf to mask self‑loops
        """
        B,N,C = node_embeds.shape
        # fuse coordinate positional encoding
        pos_emb = self.pos_encoder(node_coords)
        x = node_embeds + pos_emb

        # multi‑layer graph Transformer interaction
        for layer in self.layers:
            x = layer(x)

        # iterate over all (i,j) node pairs to predict edges
        edge_logits = torch.zeros((B,N,N), device=x.device)
        for i in range(N):
            feat_i = x[:,i:i+1,:].expand(-1,N,-1) # [B,N,C]
            logits_i = self.edge_predictor(feat_i, x) # [B,N]
            edge_logits[:,i,:] = logits_i

        # mask diagonal: forbid self‑connections, set to ‑inf (sigmoid → 0)
        self_loop_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        edge_logits.masked_fill_(self_loop_mask, -1e9)
        return edge_logits

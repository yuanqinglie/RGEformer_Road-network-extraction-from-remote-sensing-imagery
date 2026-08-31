# ===================== SGFD=====================
class SemanticGuidedFeatureDescriptor(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1_f5 = conv1x1(embed_dim, embed_dim)
        self.conv1_f4 = conv1x1(embed_dim, embed_dim)
        self.ups_2x = UpsampleBlock(scale_factor=2)

        self.conv1_f2 = conv1x1(embed_dim, embed_dim)
        self.conv1_f3 = conv1x1(embed_dim, embed_dim)
        self.dns_2x = DownsampleBlock(scale_factor=2)

        self.pos_enc_proj = conv1x1(embed_dim, embed_dim)
        self.semantic_gate = conv1x1(embed_dim, embed_dim)
        self.conv3_sdh = conv3x3(embed_dim, embed_dim)

    def _calc_affinity_matrix(self, SDh: torch.Tensor, SDl: torch.Tensor):
        B, C, Hl, Wl = SDl.shape
        SDh_align = self.ups_2x(SDh)
        sd_h = SDh_align.flatten(2).transpose(-1, -2)  # B, N, C
        sd_l = SDl.flatten(2).transpose(-1, -2)        # B, N, C

        TA = self.semantic_gate(SDh_align).flatten(2).transpose(-1, -2)
        PA = self.pos_enc_proj(SDl).flatten(2).transpose(-1, -2)

        sim = torch.matmul(sd_l, sd_h.transpose(-1, -2)) / (C ** 0.5)
        bias = torch.matmul(PA, TA.transpose(-1, -2))
        Asd = sim + bias
        Asd = F.softmax(Asd, dim=-1)
        return Asd

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4, f5 = feats
        # SDh: high‑level semantic descriptor
        f5_up = self.ups_2x(f5)
        part1 = F.relu(self.conv1_f5(f5_up))
        part2 = F.relu(self.conv1_f4(f4))
        SDh = part1 + part2

        # SDl: low‑level detail descriptor
        f2_down = self.dns_2x(f2)
        part_a = F.relu(self.conv1_f2(f2_down))
        part_b = F.relu(self.conv1_f3(f3))
        SDl = part_a + part_b

        Asd = self._calc_affinity_matrix(SDh, SDl)

        B, C, Hl, Wl = SDl.shape
        sd_l_flat = SDl.flatten(2).transpose(-1, -2)  # B, N, C
        sd_l_recal = sd_l_flat + torch.matmul(Asd, sd_l_flat)

        SDl_prime = sd_l_recal.transpose(-1, -2).reshape(B, C, Hl, Wl)

        sdh_conv3 = self.conv3_sdh(SDh)
        sdh_up = self.ups_2x(sdh_conv3)
        Fsd = sdh_up + SDl_prime
        return Fsd
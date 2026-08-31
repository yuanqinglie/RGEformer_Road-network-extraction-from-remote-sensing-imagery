class RGEFormer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        pretrained_backbone=False,
        grid_scale=32,
        Nmax=128,
        gsnr_num_heads=8,
        gsnr_num_iter=3,
        mstr_num_heads=8,
        mstr_num_layers=3
    ):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone, output_channels=embed_dim)
        self.sgfd = SemanticGuidedFeatureDescriptor(embed_dim=embed_dim)
        self.cnqo = CNQO(embed_dim=embed_dim, grid_scale=grid_scale, Nmax=Nmax)
        self.gsnr = GSNR(embed_dim=embed_dim, num_heads=gsnr_num_heads, num_iter=gsnr_num_iter)
        self.mstr = MSTR(embed_dim=embed_dim, num_heads=mstr_num_heads, num_layers=mstr_num_layers)

    def forward(self, img:torch.Tensor):
        """
        img: [B,3,512,512]
        returns：
            refined_coords: [B,Nmax,2] refined node coordinates
            refined_embeds: [B,Nmax,256] refined node features
            edge_logits: [B,Nmax,Nmax] edge prediction logits
            Mfore, Mdens, Sfd_map: for semantic mask loss computation
            Fsd: fused multi‑scale feature map
        """
        multi_feats = self.backbone(img)
        Fsd = self.sgfd(multi_feats)
        init_coords, init_embeds, Mfore, Mdens, Sfd_map = self.cnqo(Fsd)
        refined_coords, refined_embeds = self.gsnr(init_coords, init_embeds, Fsd)
        edge_logits = self.mstr(refined_coords, refined_embeds)
        return refined_coords, refined_embeds, edge_logits, Mfore, Mdens, Sfd_map, Fsd


# ===================== Unit Test =====================
if __name__ == "__main__":
    # ⚠️ SGFD global affinity matrix consumes massive GPU memory, test with batch_size=1
    batch_size = 1
    H,W = 512, 512
    input_img = torch.randn(batch_size, 3, H, W)

    model = RGEFormer(
        embed_dim=256,
        pretrained_backbone=False,
        grid_scale=32,
        Nmax=128,
        gsnr_num_iter=3,
        mstr_num_layers=3
    )

    refined_coords, refined_embeds, edge_logits, Mfore, Mdens, Sfd_map, Fsd = model(input_img)

    print(f"Input image shape: {input_img.shape}")
    print(f"Fsd fused feature map shape: {Fsd.shape}")
    print(f"refined_coords shape: {refined_coords.shape}")
    print(f"refined_embeds shape: {refined_embeds.shape}")
    print(f"edge_logits (adjacency matrix logits) shape: {edge_logits.shape}")
    print(f"Mfore shape: {Mfore.shape}")
    print("\modules executed successfully!")
# sam_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from segment_anything.modeling.image_encoder import ImageEncoderViT


def conv1x1(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)


class Backbone(nn.Module):
    """
    SAM ViT Image Encoder Backbone for RGEFormer
    Native SAM‑ViT only produces 1/16 feature map.
    Other scales are obtained via bilinear interpolation to match multi‑scale interface.
    Output order: [f1(1/2), f2(1/4), f3(1/8), f4(1/16), f5(1/32)]
    """
    def __init__(
        self,
        pretrained: bool = False,
        output_channels: int = 256,
        sam_vit_type: str = "vit_b",
        sam_checkpoint_path: str = None
    ):
        super().__init__()

        # Build SAM Image Encoder (ViT‑B as default)
        if sam_vit_type == "vit_b":
            self.image_encoder = ImageEncoderViT(
                depth=12,
                embed_dim=768,
                num_heads=12,
                patch_size=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=torch.nn.LayerNorm,
            )
            encoder_out_dim = 256  # SAM neck output channel
        elif sam_vit_type == "vit_l":
            self.image_encoder = ImageEncoderViT(
                depth=24,
                embed_dim=1024,
                num_heads=16,
                patch_size=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=torch.nn.LayerNorm,
            )
            encoder_out_dim = 256
        elif sam_vit_type == "vit_h":
            self.image_encoder = ImageEncoderViT(
                depth=32,
                embed_dim=1280,
                num_heads=16,
                patch_size=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=torch.nn.LayerNorm,
            )
            encoder_out_dim = 256
        else:
            raise ValueError(f"Unsupported sam_vit_type: {sam_vit_type}")

        # Load official SAM pretrained weights
        if pretrained:
            if sam_checkpoint_path is None:
                raise RuntimeError("Please provide sam_checkpoint_path for pretrained SAM")
            state_dict = torch.load(sam_checkpoint_path, map_location="cpu")
            self.image_encoder.load_state_dict(state_dict["image_encoder"], strict=True)

        # Projection layers for each scale
        self.proj_f1 = conv1x1(encoder_out_dim, output_channels)
        self.proj_f2 = conv1x1(encoder_out_dim, output_channels)
        self.proj_f3 = conv1x1(encoder_out_dim, output_channels)
        self.proj_f4 = conv1x1(encoder_out_dim, output_channels)
        self.proj_f5 = conv1x1(encoder_out_dim, output_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: input image tensor [B,3,H,W], recommend 512×512
        Returns:
            [f1, f2, f3, f4, f5]
                f1: 1/2 resolution
                f2: 1/4 resolution
                f3: 1/8 resolution
                f4: 1/16 resolution (native SAM output)
                f5: 1/32 resolution
        """
        # native SAM output: 1/16 scale
        feat_1_16 = self.image_encoder(x)  # [B,256,H/16,W/16]

        # interpolate to get other required scales
        B, C, H16, W16 = feat_1_16.shape

        feat_1_8 = F.interpolate(feat_1_16, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat_1_4 = F.interpolate(feat_1_16, scale_factor=4.0, mode="bilinear", align_corners=False)
        feat_1_2 = F.interpolate(feat_1_16, scale_factor=8.0, mode="bilinear", align_corners=False)
        feat_1_32 = F.interpolate(feat_1_16, scale_factor=0.5, mode="bilinear", align_corners=False)

        # project each scale to unified output_channels
        f1 = self.proj_f1(feat_1_2)
        f2 = self.proj_f2(feat_1_4)
        f3 = self.proj_f3(feat_1_8)
        f4 = self.proj_f4(feat_1_16)
        f5 = self.proj_f5(feat_1_32)

        return [f1, f2, f3, f4, f5]


if __name__ == "__main__":
    # Unit test, do not need pretrained checkpoint for test
    B, C, H, W = 1, 3, 512, 512
    inp = torch.randn(B, C, H, W)
    net = Backbone(pretrained=False, output_channels=256, sam_vit_type="vit_b")
    feats = net(inp)
    for idx, ft in enumerate(feats):
        print(f"f{idx+1} shape: {ft.shape}")
    print("✅ SAM‑ViT Backbone test pass")
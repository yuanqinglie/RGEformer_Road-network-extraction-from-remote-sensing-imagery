# sam2_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from sam2.modeling.image_encoder import ImageEncoderViT


def conv1x1(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)


class Backbone(nn.Module):
    """
    SAM2 ViT Image Encoder Backbone for RGEFormer
    Native multi‑scale outputs: 1/4,1/8,1/16,1/32.
    f1(1/2) is obtained by up‑sampling native 1/4 feature.
    Output order: [f1(1/2), f2(1/4), f3(1/8), f4(1/16), f5(1/32)]
    """
    def __init__(
        self,
        pretrained: bool = False,
        output_channels: int = 256,
        sam2_vit_type: str = "vit_b",
        sam2_checkpoint_path: str = None
    ):
        super().__init__()

        # Build SAM2 image encoder
        if sam2_vit_type == "vit_b":
            self.image_encoder = ImageEncoderViT(
                depth=12,
                embed_dim=768,
                num_heads=12,
                patch_size=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=torch.nn.LayerNorm,
            )
            # SAM2‑vit_b stage output channels: 1/4:128, 1/8:256, 1/16:512, 1/32:1024
            c2, c3, c4, c5 = 128, 256, 512, 1024
        elif sam2_vit_type == "vit_l":
            self.image_encoder = ImageEncoderViT(
                depth=24,
                embed_dim=1024,
                num_heads=16,
                patch_size=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=torch.nn.LayerNorm,
            )
            c2, c3, c4, c5 = 256, 512, 1024, 2048
        else:
            raise ValueError(f"Unsupported sam2_vit_type: {sam2_vit_type}, only vit_b / vit_l available")

        self._feat_store = {}

        # Register hooks to capture real multi‑scale features from each stage
        self.image_encoder.stages[0].register_forward_hook(self._hook("s2"))   # 1/4
        self.image_encoder.stages[1].register_forward_hook(self._hook("s3"))   # 1/8
        self.image_encoder.stages[2].register_forward_hook(self._hook("s4"))   # 1/16
        self.image_encoder.stages[3].register_forward_hook(self._hook("s5"))   # 1/32

        # Projection layers for each scale
        self.proj_f1 = conv1x1(c2, output_channels)
        self.proj_f2 = conv1x1(c2, output_channels)
        self.proj_f3 = conv1x1(c3, output_channels)
        self.proj_f4 = conv1x1(c4, output_channels)
        self.proj_f5 = conv1x1(c5, output_channels)

        # load official sam2 weights
        if pretrained:
            if sam2_checkpoint_path is None:
                raise RuntimeError("Set sam2_checkpoint_path when pretrained=True")
            ckpt = torch.load(sam2_checkpoint_path, map_location="cpu")
            self.image_encoder.load_state_dict(ckpt["image_encoder"], strict=True)

    def _hook(self, name):
        def hook_fn(module, inp, out):
            self._feat_store[name] = out
        return hook_fn

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: input tensor [B,3,H,W], recommend 512×512
        Returns:
            [f1(1/2), f2(1/4), f3(1/8), f4(1/16), f5(1/32)]
        """
        self._feat_store.clear()
        _ = self.image_encoder(x)

        feat_1_4 = self._feat_store["s2"]
        feat_1_8 = self._feat_store["s3"]
        feat_1_16 = self._feat_store["s4"]
        feat_1_32 = self._feat_store["s5"]

        # Only f1(1/2) is upsampled from native 1/4 feature
        feat_1_2 = F.interpolate(feat_1_4, scale_factor=2.0, mode="bilinear", align_corners=False)

        # Project every scale to unified output_channels
        f1 = self.proj_f1(feat_1_2)
        f2 = self.proj_f2(feat_1_4)
        f3 = self.proj_f3(feat_1_8)
        f4 = self.proj_f4(feat_1_16)
        f5 = self.proj_f5(feat_1_32)

        return [f1, f2, f3, f4, f5]


if __name__ == "__main__":
    # Unit test (no pretrained weights required)
    B, C, H, W = 1, 3, 512, 512
    inp = torch.randn(B, C, H, W)
    net = Backbone(pretrained=False, output_channels=256, sam2_vit_type="vit_b")
    feats = net(inp)
    for idx, ft in enumerate(feats):
        print(f"f{idx+1} shape: {ft.shape}")
    print("✅ SAM2‑ViT Backbone test pass")
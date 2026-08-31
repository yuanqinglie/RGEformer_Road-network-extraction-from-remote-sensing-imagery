# convnext_t_backbone.py
import torch
import torch.nn as nn
from torchvision import models
from typing import List


def conv1x1(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)


class Backbone(nn.Module):
    """ConvNeXt‑Tiny backbone, output 5‑scale features [f1,f2,f3,f4,f5]"""
    def __init__(self, pretrained: bool = False, output_channels: int = 256):
        super().__init__()
        cnx = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.features = cnx.features
        self._feat_store = {}

        self.features[0].register_forward_hook(self._hook("f1"))
        self.features[1].register_forward_hook(self._hook("f2"))
        self.features[3].register_forward_hook(self._hook("f3"))
        self.features[5].register_forward_hook(self._hook("f4"))
        self.features[7].register_forward_hook(self._hook("f5"))

        self.proj_f1 = conv1x1(96, output_channels)
        self.proj_f2 = conv1x1(96, output_channels)
        self.proj_f3 = conv1x1(192, output_channels)
        self.proj_f4 = conv1x1(384, output_channels)
        self.proj_f5 = conv1x1(768, output_channels)

    def _hook(self, name):
        def hook_fn(module, inp, out):
            self._feat_store[name] = out
        return hook_fn

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._feat_store.clear()
        _ = self.features(x)

        f1 = self.proj_f1(self._feat_store["f1"])
        f2 = self.proj_f2(self._feat_store["f2"])
        f3 = self.proj_f3(self._feat_store["f3"])
        f4 = self.proj_f4(self._feat_store["f4"])
        f5 = self.proj_f5(self._feat_store["f5"])
        return [f1, f2, f3, f4, f5]


if __name__ == "__main__":
    B, C, H, W = 1, 3, 512, 512
    inp = torch.randn(B, C, H, W)
    net = Backbone(pretrained=False, output_channels=256)
    feats = net(inp)
    for idx, ft in enumerate(feats):
        print(f"f{idx+1} shape: {ft.shape}")
    print("✅ ConvNeXt‑Tiny Backbone test pass")
# backbone_test.py
import importlib
from typing import Optional, Dict
import torch
import torch.nn as nn

SUPPORTED_BACKBONES = [
    "resnet50",
    "mobilenetv3",
    "efficientnet_b4",
    "swin_t",
    "convnext_t",
    "sam_v1",
    "sam2"
]

# Mapping backbone name to corresponding python module file
_NAME_TO_MODULE = {
    "resnet50": "resnet_backbone",
    "mobilenetv3": "mobilenetv3_backbone",
    "efficientnet_b4": "efficientnet_b4_backbone",
    "swin_t": "swin_t_backbone",
    "convnext_t": "convnext_t_backbone",
    "sam_v1": "sam_backbone",
    "sam2": "sam2_backbone",
}


def get_backbone(
    backbone_name: str,
    pretrained: bool = False,
    output_channels: int = 256,
    **kwargs
) -> nn.Module:
    """
    Backbone factory function.
    Instantiate target backbone by name string.
    Args:
        backbone_name: available options: resnet50 / mobilenetv3 / efficientnet_b4 / swin_t / convnext_t / sam_v1 / sam2
        pretrained: whether to load pretrained weights
        output_channels: unified output channel dimension for multi‑scale features
        kwargs: extra parameters for SAM / SAM2:
            sam_v1: sam_vit_type, sam_checkpoint_path
            sam2: sam2_vit_type, sam2_checkpoint_path
    Returns:
        Backbone module, returns multi‑scale feature list [f1,f2,f3,f4,f5]
    """
    backbone_name = backbone_name.lower()
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}\n"
            f"Supported list: {SUPPORTED_BACKBONES}"
        )

    module_name = _NAME_TO_MODULE[backbone_name]
    mod = importlib.import_module(module_name)
    backbone = mod.Backbone(
        pretrained=pretrained,
        output_channels=output_channels,
        **kwargs
    )
    return backbone


if __name__ == "__main__":
    # Unit test: initialization & forward pass, no pretrained weights required
    B, C, H, W = 1, 3, 512, 512
    dummy_input = torch.randn(B, C, H, W)

    # Skip sam_v1 / sam2 in local quick test (require checkpoint files)
    test_list = ["resnet50", "mobilenetv3", "swin_t"]
    for name in test_list:
        print(f"\n==== Test backbone: {name} ====")
        net = get_backbone(name, pretrained=False, output_channels=256)
        feats = net(dummy_input)
        assert len(feats) == 5, f"{name} must output exactly 5 feature maps"
        for i, ft in enumerate(feats):
            print(f"  f{i+1}: {ft.shape}")
    print("\n Backbone factory test pass")
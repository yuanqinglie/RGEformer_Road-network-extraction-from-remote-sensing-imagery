# ===================== 2.1 ResNet Backbone Network =====================
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = False, output_channels: int = 256):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.proj_f1 = conv1x1(64, output_channels)
        self.proj_f2 = conv1x1(256, output_channels)
        self.proj_f3 = conv1x1(512, output_channels)
        self.proj_f4 = conv1x1(1024, output_channels)
        self.proj_f5 = conv1x1(2048, output_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(x)
        x = self.maxpool(f1)
        f2 = self.layer1(x)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)

        f1 = self.proj_f1(f1)
        f2 = self.proj_f2(f2)
        f3 = self.proj_f3(f3)
        f4 = self.proj_f4(f4)
        f5 = self.proj_f5(f5)
        return [f1, f2, f3, f4, f5]
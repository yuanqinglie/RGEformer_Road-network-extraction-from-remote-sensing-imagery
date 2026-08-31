
import argparse
import torch
from PIL import Image
from torchvision import transforms

from rgeformer import RGEFormer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = get_args()
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    image = Image.open(args.image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(args.device)

    model = RGEFormer(backbone_name=args.backbone, pretrained_backbone=False).to(args.device)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
    print(f"Model output shape: {output.shape}")
    # TODO: add visualization / save prediction result

if __name__ == "__main__":
    main()
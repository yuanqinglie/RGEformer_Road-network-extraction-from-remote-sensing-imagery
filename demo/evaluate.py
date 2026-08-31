
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from rgeformer import RGEFormer
from utils.dataset import CustomDataset
from utils.metrics import AverageMeter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = get_args()
    transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    val_dataset = CustomDataset(args.data_root, split="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = RGEFormer(backbone_name=args.backbone, pretrained_backbone=False).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    meter = AverageMeter()
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(args.device)
            out = model(img)
            # TODO: compute your evaluation metric
            meter.update(0.0)
    print(f"Evaluation finished, metric avg: {meter.avg:.4f}")

if __name__ == "__main__":
    main()

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from rgeformer import RGEFormer
from utils.misc import set_seed, load_yaml_config, save_checkpoint
from utils.dataset import CustomDataset
from utils.metrics import AverageMeter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e‑4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./output")
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    backbone_kwargs = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)
        args.backbone = cfg.get("backbone", args.backbone)
        backbone_kwargs = cfg.get("backbone_kwargs", {})

    # build model
    model = RGEFormer(
        backbone_name=args.backbone,
        pretrained_backbone=True,
        backbone_kwargs=backbone_kwargs
    ).to(args.device)

    # dataset & dataloader
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    train_dataset = CustomDataset(args.data_root, split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # TODO: replace with your real loss function

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        for batch in train_loader:
            images = batch["image"].to(args.device)
            # TODO: get ground‑truth labels
            outputs = model(images)
            loss = criterion(outputs, torch.zeros_like(outputs)) # placeholder

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {loss_meter.avg:.4f}")
        save_checkpoint(model, optimizer, epoch+1, os.path.join(args.save_dir, f"ckpt_epoch{epoch+1}.pth"))

if __name__ == "__main__":
    main()
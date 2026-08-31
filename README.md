# RGEFormer: Road network extraction integrating graph representation and Transformer‑based query optimization from remote sensing imagery

> Qinglie Yuan

> ✉️ yuanqinglie@pzhu.edu.cn

**Official PyTorch implementation for RGEFormer**, an end‑to‑end road network graph extraction framework for remote‑sensing images.
The code is still being continuously optimized. Welcome to provide improvement suggestions.

## 📌 Framework Overview

RGEFormer is an end‑to‑end CNN‑Transformer hybrid framework for remote‑sensing road network graph extraction. Given an input remote‑sensing image, the model directly outputs:
- Road node coordinates `[B, N, 2]`
- Node classification score (real road node / background node)
- Road topology adjacency matrix `[B, N, N]` (road‑edge connection probability)
- Road foreground mask & road density mask

Compared with segmentation‑based road extraction pipelines, RGEFormer avoids heavy topology‑repair post‑processing, directly outputs graph‑form road networks (nodes + edges).

### 🔧 Main Modules

#### 1. Semantic‑Guided Multi‑scale Descriptor (SGFD)
Backbone feature extraction module built on ResNet‑50.
- Uses **gated positional affinity** to recalibrate multi‑scale cross‑level features.
- Suppresses complex background noise (vegetation, building shadows, occlusions).
- Output recalibrated multi‑scale feature maps for subsequent node query generation.

#### 2. Candidate Node Query Optimizer (CNQO)
Generate adaptive candidate road node queries.
- Fuses road foreground confidence map `Mfore` and topology‑guided kernel density map `Mdens`.
- Adaptively distribute node queries over high‑probability road regions.
- Loss: Dice loss for foreground mask `Mfore`; Quality‑Focal Loss for density map `Mdens`.

#### 3. Geometric‑Semantic Constrained Node Refinement Module (GSNR)
Iteratively refine node coordinates and node feature embeddings.
- Deformable attention enhanced with directional geometric reference.
- Introduce K‑nearest neighbor context to inject road geometry prior.
- Optimize node position offset iteratively.
- Supervised by node position regression loss $\mathcal{L}_{node\_pos}$ and node binary classification loss $\mathcal{L}_{node\_cls}$.

#### 4. Map‑Structure Transformer (MSTR) Topology Reconstruction Decoder
Graph‑Transformer decoder for road topology reconstruction.
- Coordinate positional encoding inject geometric information into node features.
- Multi‑layer graph transformer for node‑to‑node semantic & geometric interaction.
- Edge prediction head outputs pairwise road‑connection logits, generating adjacency matrix.
- Regularized by adjacency fitting loss, mutual‑KNN symmetry constraint loss, node‑degree constraint loss.

> Multi‑task weighted joint loss:
$$
\mathcal{L}_{total}=\lambda_1 \mathcal{L}_{CNQO} + \lambda_2 \mathcal{L}_{node\_pos} + \lambda_3 \mathcal{L}_{node\_cls} + \lambda_4 \mathcal{L}_{sym} + \lambda_5 \mathcal{L}_{ndg}
$$

---

## 🧪 Environment & Installation
### Environment Requirements

| Package | Version |
|---|---|
| Python | >=3.9 |
| PyTorch | >=2.0.0 |
| Torchvision | >=0.15.0 |
| CUDA | >=11.7 |
| opencv‑python | >=4.7 |
| numpy | >=1.23 |
| scipy | >=1.10 |
| matplotlib | >=3.7 |
| tqdm | latest |

### Install
```bash
git clone https://github.com/yuanql‑pzhu/RGEFormer.git
cd RGEFormer

# create conda environment
conda create -n rgeformer python=3.10
conda activate rgeformer

# install dependencies
pip install -r requirements.txt
```

> Install PyTorch matching your CUDA version:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📂 Dataset Preparation
We use three public remote‑sensing road network datasets.
Dataset folder structure:
```
datasets/
├─ train/
│   ├─ img/          # remote‑sensing RGB images
│   ├─ mask_fore/    # road foreground binary mask Mfore
│   ├─ mask_dens/    # road kernel density map Mdens
│   ├─ node_coord/   # ground‑truth node coordinates (*.npy)
│   ├─ node_cls/     # ground‑truth node classification label (*.npy)
│   └─ adj/          # ground‑truth adjacency matrix (*.npy)
├─ val/
└─ test/
```

Each sample contains:
1. Input image: `(3,512,512)` RGB
2. `Mfore` road foreground mask: `(1,512,512)` binary 0‑1
3. `Mdens` road density map: `(1,512,512)` float 0‑1
4. `gt_node_coords`: `(Nmax,2)` normalized coordinate `[-1,1]`
5. `gt_node_cls`: `(Nmax,)` binary label 0/1
6. `gt_adj_matrix`: `(Nmax,Nmax)` binary adjacency matrix.

> Dataset conversion script: `tools/preprocess_dataset.py`

---

## 🚀 Training & Evaluation
### Train
```bash
python train.py \
  --batch_size 2 \
  --epochs 150 \
  --lr 1e-4 \
  --n_max_node 128 \
  --knn_k 16 \
  --lambda1 1.0 \
  --lambda2 2.0 \
  --lambda3 2.0 \
  --lambda4 1.5 \
  --lambda5 1.0 \
  --dataset_root ./datasets
```

Hyper‑parameters:
- $\lambda_1=1.0,\lambda_2=2.0,\lambda_3=2.0,\lambda_4=1.5,\lambda_5=1.0$ (initial empirical value, grid‑search on validation set for optimal weights).
- `n_max_node`: maximum number of candidate road nodes.
- `knn_k`: K‑nearest neighbor number for symmetry loss.

### Evaluation / Test
```bash
python test.py \
  --checkpoint ./weights/best_model.pth \
  --dataset_root ./datasets/test \
  --edge_threshold 0.5
```

Evaluation metrics:
- `mTOPO‑F1`: Topology‑aware F1‑score for road network graph.
- `mAPLS`: Mean Average Path‑Length Score for topology connectivity quality.

### Inference Demo (Single image)
```python
from demo import inference_image

model_weight = "./weights/best_model.pth"
img_path = "./demo/test_01.png"
pred_coords, pred_adj_matrix, pred_mask_fore = inference_image(
    model_weight, img_path, edge_threshold=0.5
)
```

---

## 👁️ Visualization
Visualization scripts located in `tools/visualize.py`
- Draw input remote‑sensing image + predicted road nodes
- Draw predicted road topology graph (nodes + edges)
- Visualize foreground mask `Mfore` and density mask `Mdens` heatmap

```bash
python tools/visualize.py \
  --img ./demo/test_01.png \
  --ckpt ./weights/best_model.pth \
  --save_dir ./output_vis
```

Output visualization files:
- `*_img.png`: input remote‑sensing image
- `*_nodes.png`: visualized road nodes
- `*_graph.png`: visualized predicted road topology graph
- `*_mask.png`: Mfore & Mdens heatmap.

---

## 📉 Loss Function
Multi‑task joint loss implementation in `loss/rge_loss.py`.

Total loss:
$$
\mathcal{L}_{total}=\lambda_1 \mathcal{L}_{CNQO} + \lambda_2 \mathcal{L}_{node\_pos} + \lambda_3 \mathcal{L}_{node\_cls} + \lambda_4 \mathcal{L}_{sym} + \lambda_5 \mathcal{L}_{ndg}
$$

1. $\mathcal{L}_{CNQO}$: DiceLoss(`Mfore`) + Quality‑FocalLoss(`Mdens`)
2. $\mathcal{L}_{node\_pos}$: L1 loss for node coordinate regression
3. $\mathcal{L}_{node\_cls}$: BCEWithLogitsLoss for road node binary classification
4. $\mathcal{L}_{adj}$: Frobenius‑norm loss for adjacency matrix fitting
5. $\mathcal{L}_{sym}$: Mutual‑KNN pair symmetry constraint loss, suppress one‑way spurious edges
6. $\mathcal{L}_{ndg}$: Node‑degree constraint loss, penalize degree mismatch between prediction and ground‑truth.

```python
from loss.rge_loss import RGEFormerLoss
criterion = RGEFormerLoss(
    lambda1=1.0,lambda2=2.0,lambda3=2.0,lambda4=1.5,lambda5=1.0,
    knn_k=16
)
```

---

## 📊 Main Quantitative Results

| Method | mTOPO‑F1 (%) | mAPLS (%) |
|---|---|---|
| SOTA‑baseline‑1 | 78.14 | 66.21 |
| SOTA‑baseline‑2 | 81.37 | 69.43 |
| **RGEFormer(Ours)** | **85.62** | **74.77** |

> RGEFormer achieves strong robustness for shadow, occlusion and complex road geometry.

---

## 📎 Citation
If you find this repository helpful for your research, please cite our paper:
```bibtex
@article{yuan2026rgeformer,
  title={Road network extraction integrating graph representation and Transformer‑based query optimization from remote sensing imagery},
  author={Yuan, Qinglie},
  journal={XXX},
  year={2026},
  institution={Panzhihua University}
}
```

---

## 📋 TODO List
- [x] Release full model & loss code
- [x] Training & evaluation pipeline
- [x] Visualization script
- [ ] Release pre‑trained weights
- [ ] Support custom dataset

## License
This project is released under the MIT License.

> ⚠️ This repository is for academic research only.

---

## 📁 Project Directory Structure
```
RGEFormer/
├─ assets/                # figures for readme
├─ configs/               # training config yaml
├─ datasets/              # dataset folder
├─ loss/
│   └─ rge_loss.py        # multi‑task loss function
├─ models/
│   ├─ backbone.py        # ResNet‑50 + SGFD module
│   ├─ cnqo.py            # CNQO candidate node optimizer
│   ├─ gsnr.py            # geometric‑semantic node refinement module
│   ├─ mstr.py            # Map‑Structure Transformer
│   └─ rgeformer.py       # full RGEFormer model wrapper
├─ tools/
│   ├─ preprocess_dataset.py
│   └─ visualize.py       # visualization script
├─ weights/               # model checkpoint
├─ train.py
├─ test.py
├─ demo.py
└─ requirements.txt
```

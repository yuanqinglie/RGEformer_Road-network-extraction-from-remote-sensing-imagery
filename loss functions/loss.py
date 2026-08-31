import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for Mfore road foreground mask"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred: [B,1,H,W] probability map (already sigmoid normalized)
        gt: [B,1,H,W] binary 0/1 mask
        """
        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice


class QualityFocalLoss(nn.Module):
    """Quality‑Focal Loss for Mdens density map supervision"""
    def __init__(self, gamma=2.0, beta=2.0):
        super().__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred: [B,1,H,W], sigmoid probability output
        gt: [B,1,H,W], density map with value range [0~1]
        """
        pt = pred
        weight = torch.abs(gt - pt).pow(self.gamma)
        loss = weight * F.binary_cross_entropy(pred, gt, reduction="none")
        return torch.mean(loss)


class RGEFormerLoss(nn.Module):
    def __init__(
        self,
        lambda1=1.0,    # weight for L_CNQO
        lambda2=2.0,    # weight for L_node_pos
        lambda3=2.0,    # weight for L_node_cls
        lambda4=1.5,    # weight for L_sym
        lambda5=1.0,    # weight for L_ndg
        lambda_adj=1.0, # weight for L_adj adjacency matrix fitting loss, topology loss in paper main body, not written in formula (20)
        knn_k=16,       # K(i), number of KNN neighbors used for mutual‑KNN pair calculation in L_sym
        eps=1e-6
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.lambda_adj = lambda_adj
        self.knn_k = knn_k
        self.eps = eps

        self.dice_loss = DiceLoss(eps=eps)
        self.qfl_loss = QualityFocalLoss()

    @staticmethod
    def calc_mutual_knn_mask(coords: torch.Tensor, k: int):
        """
        Compute mask for mutual‑KNN set P, corresponds to formula (22) $\mathcal{P} = \{(i,j)|j\in\mathcal{K}(i),i\in\mathcal{K}(j)\}$
        coords: [B,N,2] normalized node coordinates
        return mutual_mask: [B,N,N] bool mask, True means pair (i,j) belongs to set P
        """
        B, N, _ = coords.shape
        # Compute pairwise Euclidean distance
        dist = torch.cdist(coords, coords, p=2)  # [B,N,N]
        # Get top‑k nearest neighbor indices, exclude self‑loop
        knn_idx = torch.topk(dist, k=k+1, dim=-1, largest=False).indices
        knn_idx = knn_idx[..., 1:]  # [B,N,k] remove self index, keep k neighbors

        # Build boolean mask for K(i) neighbor set [B,N,N]
        knn_mask = torch.zeros((B, N, N), device=coords.device, dtype=torch.bool)
        for b in range(B):
            for i in range(N):
                knn_mask[b, i, knn_idx[b, i]] = True
        # Mutual‑KNN condition: j ∈ K(i) AND i ∈ K(j)
        mutual_mask = torch.logical_and(knn_mask, knn_mask.transpose(-1, -2))
        return mutual_mask

    def forward(
        self,
        # Model outputs from RGEFormer forward pass
        Mfore: torch.Tensor,
        Mdens: torch.Tensor,
        refined_coords: torch.Tensor,
        refined_embeds: torch.Tensor,
        edge_logits: torch.Tensor,
        # Ground‑Truth labels
        gt_mfore: torch.Tensor,        # [B,1,H,W] road foreground binary mask 0‑1
        gt_mdens: torch.Tensor,        # [B,1,H,W] road density map 0‑1
        gt_node_coords: torch.Tensor,  # [B,N,2] ground‑truth node coordinates
        gt_node_cls: torch.Tensor,      # [B,N] node binary classification label 0/1 indicating real road node
        gt_adj_matrix: torch.Tensor    # [B,N,N] ground‑truth adjacency matrix 0/1
    ):
        B, Nnode, _ = refined_coords.shape

        # ========== 1. L_CNQO  λ1 ==========
        mfore_prob = torch.sigmoid(Mfore)
        mdens_prob = torch.sigmoid(Mdens)
        loss_mfore = self.dice_loss(mfore_prob, gt_mfore)
        loss_mdens = self.qfl_loss(mdens_prob, gt_mdens)
        loss_cnqo = loss_mfore + loss_mdens

        # ========== 2. L_node_pos node coordinate regression with L1 loss λ2 ==========
        loss_node_pos = F.l1_loss(refined_coords, gt_node_coords)

        # ==========3. L_node_cls node binary classification BCE loss λ3 ==========
        # Note: refined_embeds has no classification head. You need to add an extra node_cls_head, output node_cls_logits:[B,N]
        # Hint: add self.node_cls_head = nn.Linear(embed_dim,1) inside RGEFormer/MSTR model
        node_cls_logits = self.node_cls_head(refined_embeds).squeeze(-1) # [B,N]
        loss_node_cls = F.binary_cross_entropy_with_logits(node_cls_logits, gt_node_cls.float())

        # ========== Topology related loss, compute edge probability S_ij ==========
        S_ij = torch.sigmoid(edge_logits) # [B,N,N] edge score S_ij

        # -------- L_adj Eq.(21) adjacency matrix Frobenius norm loss --------
        diff_adj = S_ij - gt_adj_matrix
        fro_norm = torch.linalg.norm(diff_adj, ord="fro", dim=(-2,-1)) # [B]
        loss_adj = torch.mean(fro_norm** 2) / (Nnode**2)

        # -------- L_sym Eq.(22) mutual‑KNN symmetry constraint λ4 --------
        mutual_mask = self.calc_mutual_knn_mask(refined_coords, k=self.knn_k) # [B,N,N]
        if torch.any(mutual_mask):
            sym_diff = torch.abs(S_ij - S_ij.transpose(-1,-2)) # |S_ij‑S_ji|
            selected = sym_diff[mutual_mask]
            loss_sym = torch.mean(selected)
        else:
            loss_sym = 0.0 * loss_cnqo # zero loss when no mutual‑KNN pairs, keep gradient flow valid

        # -------- L_ndg Eq.(23) node degree constraint λ5 --------
        deg_pred = torch.sum(S_ij, dim=-1) # [B,Nnode] predicted node degree deg(A_pred,i)
        deg_gt = torch.sum(gt_adj_matrix, dim=-1) # [B,Nnode] ground‑truth node degree deg(A_gt,i)
        loss_ndg = torch.mean(torch.abs(deg_pred - deg_gt))

        # ========== Total Loss Eq.(20) + supplementary topology loss L_adj ==========
        loss_total = (
            self.lambda1 * loss_cnqo
            + self.lambda2 * loss_node_pos
            + self.lambda3 * loss_node_cls
            + self.lambda4 * loss_sym
            + self.lambda5 * loss_ndg
            + self.lambda_adj * loss_adj
        )

        loss_dict = {
            "loss_total": loss_total,
            "loss_cnqo": loss_cnqo,
            "loss_mfore_dice": loss_mfore,
            "loss_mdens_qfl": loss_mdens,
            "loss_node_pos": loss_node_pos,
            "loss_node_cls": loss_node_cls,
            "loss_adj": loss_adj,
            "loss_sym": loss_sym,
            "loss_ndg": loss_ndg
        }
        return loss_total, loss_dict
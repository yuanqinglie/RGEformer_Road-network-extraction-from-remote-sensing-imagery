def postprocess_topology(edge_logits, threshold=0.5):
    """
    edge_logits: [B,N,N] raw logits output from model
    threshold: binarization threshold for edge probability
    return pred_adj: [B,N,N] 0‑1 binary predicted adjacency matrix
    """
    S_ij = torch.sigmoid(edge_logits)
    pred_adj = (S_ij > threshold).to(torch.float32)
    # Force diagonal entries to zero, disable self‑loops
    N = pred_adj.shape[-1]
    eye = torch.eye(N, device=pred_adj.device).unsqueeze(0)
    pred_adj = pred_adj * (1-eye)
    return pred_adj
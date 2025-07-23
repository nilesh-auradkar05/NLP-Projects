import torch
import torch.nn.functional as F

def calculate_metrics(logits, targets):
    """Calculates loss, accuracy, and perplexity score for a batch."""
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    loss = F.cross_entropy(logits_flat, targets_flat)
    perplexity = torch.exp(loss)

    active_logits = logits_flat[targets_flat != -100]
    active_targets = targets_flat[targets_flat != -100]
    if active_targets.numel() > 0:
        predicted_labels = torch.argmax(active_logits, dim=1)
        accuracy = (predicted_labels == active_targets).sum().float() / active_targets.numel()
    else:
        accuracy = torch.tensor(0.0)

    return loss, accuracy, perplexity
import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    result = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_f1": f1_score(all_targets.numpy(), all_preds.numpy(), average="macro"),
    }
    if return_predictions:
        result["y_true"] = all_targets.numpy()
        result["y_pred"] = all_preds.numpy()
    return result

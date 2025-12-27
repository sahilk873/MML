import torch

from kg_project.eval.metrics import compute_metrics, determine_threshold


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True)


def forward_batch(model, batch, device):
    if "features" in batch:
        return model(move_to_device(batch["features"], device))
    return model(
        move_to_device(batch["c1"], device),
        move_to_device(batch["c2"], device),
        move_to_device(batch["target"], device),
    )


def run_epoch(model, loader, criterion, optimizer, device):
    total_loss = 0.0
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        logits = forward_batch(model, batch, device)
        labels = move_to_device(batch["label"], device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch["label"])
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    logits_list = []
    labels_list = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, device)
            labels = move_to_device(batch["label"], device)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(batch["label"])
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    loss = total_loss / len(loader.dataset)
    probs = torch.sigmoid(logits)
    return {
        "loss": loss,
        "probs": probs,
        "logits": logits,
        "labels": labels,
    }


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    max_epochs=30,
    patience=5,
):
    best_score = -float("inf")
    best_state = None
    wait = 0
    best_threshold = 0.5
    train_loss = 0.0
    for epoch in range(max_epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_eval = evaluate(model, val_loader, criterion, device)
        val_probs = val_eval["probs"].numpy()
        val_labels = val_eval["labels"].numpy()
        threshold = determine_threshold(val_labels, val_probs)
        val_metrics = compute_metrics(val_labels, val_probs, threshold)
        roc_auc = val_metrics.get("roc_auc") or 0.0
        if roc_auc > best_score:
            best_score = roc_auc
            best_state = model.state_dict()
            best_threshold = threshold
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = evaluate(model, val_loader, criterion, device)
    test_eval = evaluate(model, test_loader, criterion, device)
    val_metrics = compute_metrics(
        val_eval["labels"].numpy(), val_eval["probs"].numpy(), best_threshold
    )
    test_metrics = compute_metrics(
        test_eval["labels"].numpy(), test_eval["probs"].numpy(), best_threshold
    )
    return {
        "best_threshold": best_threshold,
        "train_loss": train_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_probs": test_eval["probs"].numpy(),
        "test_labels": test_eval["labels"].numpy(),
    }

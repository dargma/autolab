"""Knowledge distillation for ternary networks.

Train a small ternary student to mimic a large full-precision teacher.
KD loss = alpha * KL(student/T || teacher/T) * T^2 + (1-alpha) * CE(student, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from . import models as model_registry
from . import data as data_factory


def distill_train(
    teacher_model,
    student_model,
    dataset_name="MNIST",
    epochs=20,
    lr=0.003,
    temperature=4.0,
    alpha=0.7,
    label_smoothing=0.05,
    augmentation=True,
    batch_size=128,
    weight_decay=1e-4,
    seed=42,
):
    """Train student via knowledge distillation from teacher.

    Args:
        teacher_model: Pre-trained full-precision teacher (frozen)
        student_model: Ternary student to train
        temperature: KD temperature (higher = softer distributions)
        alpha: Weight for KD loss vs hard label loss
    Returns:
        best_accuracy, student_model
    """
    torch.manual_seed(seed)
    teacher_model.eval()

    # Data
    if augmentation and dataset_name in data_factory.DATASETS:
        ds_cls, mean, std, ch, sz, nc = data_factory.DATASETS[dataset_name]
        aug_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = ds_cls("./data", train=True, download=True, transform=aug_transform)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        train_loader, _ = data_factory.get_loaders(
            dataset_name, batch_size_train=batch_size)

    _, test_loader = data_factory.get_loaders(dataset_name, batch_size_test=256)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher_model(batch_x)

            # Student forward
            student_logits = student_model(batch_x)

            # KD loss: KL divergence on softened distributions
            # KL(student || teacher) with temperature scaling
            soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
            soft_student = F.log_softmax(student_logits / temperature, dim=1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            kd_loss = kd_loss * (temperature ** 2)  # scale by T^2

            # Hard label loss
            ce_loss = F.cross_entropy(student_logits, batch_y,
                                       label_smoothing=label_smoothing)

            # Combined loss
            loss = alpha * kd_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate
        student_model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                preds = student_model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in student_model.state_dict().items()}

        avg_loss = epoch_loss / n_batches
        print(f"  ep{epoch+1:2d}: acc={acc:.4f} best={best_acc:.4f} loss={avg_loss:.4f}")

    # Restore best
    if best_state:
        student_model.load_state_dict(best_state)

    return best_acc, student_model

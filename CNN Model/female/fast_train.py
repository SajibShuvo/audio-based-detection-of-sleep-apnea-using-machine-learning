import os
import glob
import bisect
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ==========================
# Config
# ==========================
scratch/sshuvo13/project_shared_folder_bspml_1/segments_30s/features/male/mel_spectrum
MEL_DIR = "/scratch/sshuvo13/project_shared_folder_bspml_1/whole_dataset_features/female/mel_spectrum"
LABEL_DIR = "/scratch/sshuvo13/project_shared_folder_bspml_1/rml_analysis/segment_csv_data/labels_of_each_segment"

BATCH_SIZE = 64
NUM_EPOCHS = 3          # fewer epochs, faster backup run
RANDOM_SEED = 42

# Limit how many train segments we actually use (for speed).
# If train has > MAX_TRAIN_SEGMENTS, we subsample randomly.
MAX_TRAIN_SEGMENTS = 65_000

NUM_WORKERS = 1         # use multiple workers for faster I/O
PIN_MEMORY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Dataset
# ==========================
class SleepApneaDataset(Dataset):
    def __init__(self, file_pairs):
        """
        Robust dataset loader:
        - Loads mel files with mmap_mode (fast, low RAM)
        - Loads labels fully
        - Skips corrupted files but prints them
        - Aligns length if N_mel != N_label
        - Merges apnea-related labels → 1 vs Normal → 0
        """
        self.mel_arrays = []
        self.label_arrays = []
        self.cum_lengths = []

        self.skipped_files = []
        all_label_strings = []
        self.uses_string_labels = False

        total = 0

        for mel_path, label_path in file_pairs:
            # Try loading mel
            try:
                X = np.load(mel_path, mmap_mode="r", allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Skipping MEL file (corrupted): {mel_path}")
                print(f"       Reason: {repr(e)}")
                self.skipped_files.append((mel_path, label_path))
                continue

            # Try loading labels
            try:
                y = np.load(label_path, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Skipping LABEL file (corrupted): {label_path}")
                print(f"       Reason: {repr(e)}")
                self.skipped_files.append((mel_path, label_path))
                continue

            if y.ndim > 1:
                y = np.squeeze(y)

            N_mel = X.shape[0]
            N_lab = y.shape[0]

            # Align lengths
            if N_mel != N_lab:
                N = min(N_mel, N_lab)
                print(
                    f"[WARN] Length mismatch:\n"
                    f"       mel: {mel_path} -> {N_mel}\n"
                    f"       lab: {label_path} -> {N_lab}\n"
                    f"       Using first {N} segments."
                )
                X = X[:N]
                y = y[:N]
            else:
                N = N_mel

            # Track string labels
            if y.dtype.kind in {"U", "S", "O"}:
                self.uses_string_labels = True
                all_label_strings.extend(y.tolist())

            self.mel_arrays.append(X)
            self.label_arrays.append(y)

            total += N
            self.cum_lengths.append(total)

            print(f"[INFO] Registered {mel_path}: {N} segments, shape={X.shape[1:]}")

        print(f"[INFO] Total usable segments: {total}")
        if self.skipped_files:
            print("[INFO] Skipped corrupted file pairs:")
            for mel, lbl in self.skipped_files:
                print(f"       MEL: {mel}")
                print(f"       LAB: {lbl}")

        if total == 0:
            raise RuntimeError("No valid data loaded! Check file paths or data integrity.")

        # Build binary label map
        self.label_to_int = None
        if self.uses_string_labels:
            unique = sorted(set(all_label_strings))
            print(f"[INFO] Found label classes: {unique}")

            # Normal -> 0, all others -> 1
            self.label_to_int = {lab: (0 if lab == "Normal" else 1) for lab in unique}

            print("[INFO] Binary mapping:")
            print("       Normal -> 0")
            for lab in unique:
                if lab != "Normal":
                    print(f"       {lab} -> 1 (event)")

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        local_idx = idx - prev

        X_np = self.mel_arrays[file_idx][local_idx]
        y_np = self.label_arrays[file_idx][local_idx]

        x = torch.from_numpy(np.array(X_np, copy=True)).float()

        if self.label_to_int:
            y = float(self.label_to_int[str(y_np)])
        else:
            y = float(y_np)

        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# ==========================
# Model (lighter CNN)
# ==========================
class SleepApneaCNNFast(nn.Module):
    """
    Lighter CNN:
    - fewer channels: 16, 32, 64
    - only 3 conv blocks
    - global average pooling
    Still uses 3-channel mel input and BCEWithLogits output.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(SleepApneaCNNFast, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ==========================
# Helpers
# ==========================
def find_file_pairs(mel_dir, label_dir):
    mel_files = sorted(glob.glob(os.path.join(mel_dir, "*.npy")))
    print(f"[INFO] Total mel files found: {len(mel_files)}")

    paired_files = []
    for mel_path in mel_files:
        mel_fname = os.path.basename(mel_path)
        stem = os.path.splitext(mel_fname)[0]
        label_fname = f"{stem}_segments_labels.npy"
        label_path = os.path.join(label_dir, label_fname)

        if os.path.exists(label_path):
            paired_files.append((mel_path, label_path))
        else:
            print(f"[WARN] No label file for {label_fname}, skipping mel file {mel_fname}.")

    print(f"[INFO] Usable (mel, label) pairs: {len(paired_files)}")
    return paired_files


def split_file_pairs(paired_files, seed=42):
    random.Random(seed).shuffle(paired_files)
    n_files = len(paired_files)
    n_train = int(0.7 * n_files)
    n_val = int(0.15 * n_files)
    n_test = n_files - n_train - n_val

    train_pairs = paired_files[:n_train]
    val_pairs = paired_files[n_train:n_train + n_val]
    test_pairs = paired_files[n_train + n_val:]

    print(f"[INFO] Files total: {n_files}")
    print(f"[INFO] Train files: {len(train_pairs)}")
    print(f"[INFO] Val files:   {len(val_pairs)}")
    print(f"[INFO] Test files:  {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


def fast_compute_pos_weight(dataset):
    all_labels = []
    for arr in dataset.label_arrays:
        bin_arr = np.array([dataset.label_to_int[str(x)] for x in arr])
        all_labels.append(bin_arr)

    all_labels = np.concatenate(all_labels)

    neg = (all_labels == 0).sum()
    pos = (all_labels == 1).sum()

    print(f"[INFO] Train negatives (0): {neg}")
    print(f"[INFO] Train positives (1): {pos}")

    w = neg / max(pos, 1)
    print(f"[INFO] pos_weight = {w}")
    return torch.tensor([w], dtype=torch.float32)


def subsample_dataset(dataset, max_segments, seed=42):
    """
    If len(dataset) > max_segments, randomly pick max_segments indices.
    Otherwise, return dataset as-is.
    """
    n = len(dataset)
    if max_segments is None or n <= max_segments:
        print(f"[INFO] Using full train set of {n} segments.")
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=max_segments, replace=False)
    print(f"[INFO] Subsampling train set: {n} -> {max_segments} segments.")
    return Subset(dataset, indices)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx=1, log_interval=200):
    model.train()
    running_loss = 0.0
    total_batches = len(loader)

    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            print(
                f"[Epoch {epoch_idx}] Batch {batch_idx + 1}/{total_batches} "
                f"- Loss: {loss.item():.4f}"
            )

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device, name="val", threshold=0.5):
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.float().view(-1, 1)

            logits = model(X)
            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets).ravel()
    y_scores = np.concatenate(all_probs).ravel()
    preds = (y_scores >= threshold).astype(int)

    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, digits=4)
    cm = confusion_matrix(y_true, preds)

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_scores)
    else:
        auc = None

    return acc, report, auc, cm


def plot_confusion_matrix(cm, class_names, filename, title=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    if title:
        ax.set_title(title)

    # Label cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix to {filename}")


# ==========================
# Main
# ==========================
def main():
    print(f"[INFO] Using device: {DEVICE}")

    # 1) Pair files
    paired_files = find_file_pairs(MEL_DIR, LABEL_DIR)

    # 2) Split by file
    train_pairs, val_pairs, test_pairs = split_file_pairs(paired_files, seed=RANDOM_SEED)

    # 3) Datasets
    train_dataset_full = SleepApneaDataset(train_pairs)
    val_dataset = SleepApneaDataset(val_pairs)
    test_dataset = SleepApneaDataset(test_pairs)

    # 3a) Optional subsampling of train set for speed
    train_dataset = subsample_dataset(train_dataset_full, MAX_TRAIN_SEGMENTS, seed=RANDOM_SEED)

    # 4) DataLoaders (with workers + pin_memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"[INFO] Train segments: {len(train_dataset)}")
    print(f"[INFO] Val segments:   {len(val_dataset)}")
    print(f"[INFO] Test segments:  {len(test_dataset)}")

    # 5) pos_weight from full train dataset (more accurate)
    pos_weight = fast_compute_pos_weight(train_dataset_full).to(DEVICE)

    # 6) Model, loss, optimizer
    model = SleepApneaCNNFast(in_channels=3, num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print("[INFO] Model:")
    print(model)

    # 7) Training loop
    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch_idx=epoch)

        val_acc, val_report, val_auc, val_cm = evaluate(model, val_loader, DEVICE, name="val", threshold=0.5)

        print(f"[VAL] Train Loss: {train_loss:.4f}")
        print(f"[VAL] Accuracy:   {val_acc:.4f}")
        if val_auc is not None:
            print(f"[VAL] ROC-AUC:    {val_auc:.4f}")
        print("[VAL] Classification report:")
        print(val_report)

        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[INFO] New best val AUC: {best_val_auc:.4f} at epoch {epoch}")

        plot_confusion_matrix(
            val_cm,
            class_names=["Normal (0)", "Event (1)"],
            filename=f"confusion_matrix_val_fast_epoch{epoch}.png",
            title=f"Val Confusion Matrix (Epoch {epoch})",
        )

    # 8) Load best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
        print(f"\n[INFO] Loaded best model with val AUC = {best_val_auc:.4f}")

    test_acc, test_report, test_auc, test_cm = evaluate(model, test_loader, DEVICE, name="test", threshold=0.5)

    print("\n[TEST] Final test performance (FAST model):")
    print(f"[TEST] Accuracy: {test_acc:.4f}")
    if test_auc is not None:
        print(f"[TEST] ROC-AUC:  {test_auc:.4f}")
    print("[TEST] Classification report:")
    print(test_report)

    plot_confusion_matrix(
        test_cm,
        class_names=["Normal (0)", "Event (1)"],
        filename="confusion_matrix_test_fast.png",
        title="Test Confusion Matrix (FAST model)",
    )


if __name__ == "__main__":
    main()

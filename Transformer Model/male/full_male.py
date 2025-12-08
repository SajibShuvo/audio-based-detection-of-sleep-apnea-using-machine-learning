#!/usr/bin/env python
"""
Tiny Transformer encoder for sleep apnea detection
using 30s MALE mel-spectrogram segments (full dataset).
"""

import os
import glob
import bisect
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================
# Config
# ==========================
MEL_DIR = "/scratch/sshuvo13/project_shared_folder_bspml_1/segments_30s/features/male/mel_spectrum"
LABEL_DIR = "/scratch/sshuvo13/project_shared_folder_bspml_1/rml_analysis/fixed_rml_analysis/labels_again/fixed_30s_label_outputs"

BATCH_SIZE = 32
NUM_EPOCHS = 20
RANDOM_SEED = 42

NUM_WORKERS = 2
PIN_MEMORY = True

TIME_POOL = 10  # same idea as in female script

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Pair mel and label files
# ==========================
def find_file_pairs(mel_dir, label_dir):
    mel_files = sorted(glob.glob(os.path.join(mel_dir, "*.npy")))
    print(f"[INFO] Total mel files found: {len(mel_files)}")

    pairs = []
    for mel_path in mel_files:
        mel_name = os.path.basename(mel_path)
        stem = os.path.splitext(mel_name)[0]
        label_name = f"{stem}_segments_labels.npy"
        label_path = os.path.join(label_dir, label_name)

        if os.path.exists(label_path):
            pairs.append((mel_path, label_path))
        else:
            print(f"[WARN] Missing label for {mel_name} -> {label_name}")

    print(f"[INFO] Usable (mel, label) pairs: {len(pairs)}")
    return pairs


def split_file_pairs(file_pairs, seed=42):
    rng = random.Random(seed)
    rng.shuffle(file_pairs)
    n = len(file_pairs)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    train_pairs = file_pairs[:n_train]
    val_pairs = file_pairs[n_train:n_train + n_val]
    test_pairs = file_pairs[n_train + n_val:]

    print(f"[INFO] Patient-level split (files):")
    print(f"       Train: {len(train_pairs)}")
    print(f"       Val:   {len(val_pairs)}")
    print(f"       Test:  {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


# ==========================
# Dataset
# ==========================
class MelTransformerDataset(Dataset):
    def __init__(self, file_pairs, time_pool=10):
        self.mel_arrays = []
        self.label_arrays = []
        self.cum_lengths = []

        self.time_pool = time_pool
        self.skipped = []
        self.label_to_int = None
        all_label_strings = []
        self.uses_string_labels = False

        total = 0
        for mel_path, label_path in file_pairs:
            try:
                X = np.load(mel_path, mmap_mode="r", allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Skipping MEL file: {mel_path}, reason: {repr(e)}")
                self.skipped.append((mel_path, label_path))
                continue

            try:
                y = np.load(label_path, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Skipping LABEL file: {label_path}, reason: {repr(e)}")
                self.skipped.append((mel_path, label_path))
                continue

            if y.ndim > 1:
                y = np.squeeze(y)

            N_mel = X.shape[0]
            N_lab = y.shape[0]
            if N_mel != N_lab:
                N = min(N_mel, N_lab)
                print(
                    f"[WARN] Length mismatch for {mel_path} vs {label_path}: "
                    f"mel={N_mel}, label={N_lab}. Using first {N} segments."
                )
                X = X[:N]
                y = y[:N]
            else:
                N = N_mel

            if y.dtype.kind in {"U", "S", "O"}:
                self.uses_string_labels = True
                all_label_strings.extend(y.tolist())

            self.mel_arrays.append(X)
            self.label_arrays.append(y)

            total += N
            self.cum_lengths.append(total)

            print(f"[INFO] Registered {mel_path}: {N} segments, shape={X.shape[1:]}")

        if total == 0:
            raise RuntimeError("No valid male segments loaded!")

        print(f"[INFO] Total male segments: {total}")

        if self.uses_string_labels:
            unique = sorted(set(all_label_strings))
            print(f"[INFO] Found label classes: {unique}")
            self.label_to_int = {lab: (0 if lab == "Normal" else 1) for lab in unique}
            print("[INFO] Binary label mapping:")
            print("       Normal -> 0")
            for lab in unique:
                if lab != "Normal":
                    print(f"       {lab} -> 1 (event)")
        else:
            self.label_to_int = None

    def __len__(self):
        return self.cum_lengths[-1]

    def _segment_to_sequence(self, seg):
        C, n_mels, T = seg.shape

        if self.time_pool > 1 and T >= self.time_pool:
            T_new = T // self.time_pool
            T_use = T_new * self.time_pool
            seg = seg[:, :, :T_use]
            seg = seg.reshape(C, n_mels, T_new, self.time_pool).mean(axis=-1)
        else:
            T_new = T

        seg = seg.reshape(C * n_mels, T_new)
        seg = seg.transpose(1, 0)
        return seg.astype(np.float32)

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        local_idx = idx - prev_cum

        X_np = self.mel_arrays[file_idx][local_idx]
        y_np = self.label_arrays[file_idx][local_idx]

        seq = self._segment_to_sequence(X_np)
        x = torch.from_numpy(seq)

        if self.label_to_int is not None:
            y_bin = float(self.label_to_int[str(y_np)])
        else:
            y_bin = float(y_np)

        y = torch.tensor(y_bin, dtype=torch.float32)
        return x, y


# ==========================
# Tiny Transformer (same as female)
# ==========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TinyMelTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=2000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_encoder(x)
        h = self.transformer(x)
        pooled = h.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


# ==========================
# Helpers
# ==========================
def fast_compute_pos_weight(dataset):
    all_labels = []
    for arr in dataset.label_arrays:
        if dataset.label_to_int is not None:
            bin_arr = np.array([dataset.label_to_int[str(x)] for x in arr])
        else:
            bin_arr = np.array(arr, dtype=np.float32)
        all_labels.append(bin_arr)

    all_labels = np.concatenate(all_labels)
    neg = (all_labels == 0).sum()
    pos = (all_labels == 1).sum()
    print(f"[INFO] Train negatives (0): {neg}")
    print(f"[INFO] Train positives (1): {pos}")

    w = neg / max(pos, 1)
    print(f"[INFO] pos_weight = {w:.4f}")
    return torch.tensor([w], dtype=torch.float32)


def collate_variable_length(batch):
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    feat_dim = seqs[0].shape[1]

    max_T = max(lengths)
    B = len(seqs)

    X_padded = torch.zeros(B, max_T, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        T_i = s.shape[0]
        X_padded[i, :T_i, :] = s

    y = torch.stack(labels).view(-1, 1)
    return X_padded, y


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx=1, log_interval=200):
    model.train()
    running_loss = 0.0
    total = 0

    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        running_loss += loss.item() * bs
        total += bs

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(loader):
            print(
                f"[Epoch {epoch_idx}] Batch {batch_idx + 1}/{len(loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    return running_loss / max(total, 1)


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

    print(f"[{name.upper()}] Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"[{name.upper()}] ROC-AUC: {auc:.4f}")
    print(f"[{name.upper()}] Classification report:")
    print(report)

    return acc, auc, report, cm


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

    pairs_all = find_file_pairs(MEL_DIR, LABEL_DIR)
    train_pairs, val_pairs, test_pairs = split_file_pairs(pairs_all, seed=RANDOM_SEED)

    print("\n[INFO] Building TRAIN dataset (male)...")
    train_dataset = MelTransformerDataset(train_pairs, time_pool=TIME_POOL)

    print("\n[INFO] Building VAL dataset (male)...")
    val_dataset = MelTransformerDataset(val_pairs, time_pool=TIME_POOL)

    print("\n[INFO] Building TEST dataset (male)...")
    test_dataset = MelTransformerDataset(test_pairs, time_pool=TIME_POOL)

    print(f"\n[INFO] Train segments: {len(train_dataset)}")
    print(f"[INFO] Val segments:   {len(val_dataset)}")
    print(f"[INFO] Test segments:  {len(test_dataset)}")

    pos_weight = fast_compute_pos_weight(train_dataset).to(DEVICE)

    sample_x, _ = train_dataset[0]
    feat_dim = sample_x.shape[1]
    print(f"[INFO] Sequence feature dimension: {feat_dim}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_variable_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_variable_length,
    )

    model = TinyMelTransformer(
        input_dim=feat_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print("\n[INFO] Model architecture (MALE):")
    print(model)

    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} (MALE) ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch_idx=epoch)
        print(f"[TRAIN] Loss: {train_loss:.4f}")

        val_acc, val_auc, _, val_cm = evaluate(model, val_loader, DEVICE, name="val", threshold=0.5)

        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[INFO] New best val AUC (male): {best_val_auc:.4f} at epoch {epoch}")

        plot_confusion_matrix(
            val_cm,
            class_names=["Normal (0)", "Event (1)"],
            filename=f"confusion_matrix_val_male_transformer_epoch{epoch}.png",
            title=f"Male Val Confusion Matrix (Epoch {epoch})",
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
        print(f"\n[INFO] Loaded best MALE model with val AUC = {best_val_auc:.4f}")

    print("\n[INFO] Evaluating on TEST set (male)...")
    test_acc, test_auc, _, test_cm = evaluate(model, test_loader, DEVICE, name="test", threshold=0.5)

    plot_confusion_matrix(
        test_cm,
        class_names=["Normal (0)", "Event (1)"],
        filename="confusion_matrix_test_male_transformer.png",
        title="Male Test Confusion Matrix (Tiny Transformer, 30s mel)",
    )

    print("\n[INFO] Done (male).")


if __name__ == "__main__":
    main()

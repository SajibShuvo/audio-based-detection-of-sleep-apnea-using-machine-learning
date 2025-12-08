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
matplotlib.use("Agg")  # headless for cluster runs
import matplotlib.pyplot as plt


# ==========================
# Config
# ==========================
MEL_DIR = "/scratch/sshuvo13/project_shared_folder_bspml_1/whole_dataset_features/female/mel_spectrum"
LABEL_DIR = "/scratch/sshuvo13/project_shared_folder_bspml_1/rml_analysis/segment_csv_data/labels_of_each_segment"

BATCH_SIZE = 16          # smaller for Transformer
NUM_EPOCHS = 8           # more capacity per segment, fewer patients
RANDOM_SEED = 42

NUM_WORKERS = 1
PIN_MEMORY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer hyperparams
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 3
DIM_FF = 256
DROPOUT = 0.1
TIME_SUBSAMPLE = 4   # use every 4th frame -> shorter sequence


# ==========================
# Dataset
# ==========================
class SleepApneaDataset(Dataset):
    def __init__(self, file_pairs):
        """
        Dataset for mel-spectrogram segments with binary labels.
        - mel: (N, C, n_mels, frames) float32
        - labels: strings like "Normal", "ObstructiveApnea", etc.
        We map: Normal -> 0, all other labels -> 1 (apnea event).
        """
        self.mel_arrays = []
        self.label_arrays = []
        self.cum_lengths = []

        self.skipped_files = []
        all_label_strings = []
        self.uses_string_labels = False

        total = 0

        for mel_path, label_path in file_pairs:
            # Load mel
            try:
                X = np.load(mel_path, mmap_mode="r", allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Skipping MEL file (corrupted): {mel_path}")
                print(f"       Reason: {repr(e)}")
                self.skipped_files.append((mel_path, label_path))
                continue

            # Load labels
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

            # Length alignment
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

            # Track label strings
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

        X_np = self.mel_arrays[file_idx][local_idx]   # (C, n_mels, frames)
        y_np = self.label_arrays[file_idx][local_idx]

        # Copy so it's writable
        x = torch.from_numpy(np.array(X_np, copy=True)).float()

        if self.label_to_int:
            y = float(self.label_to_int[str(y_np)])
        else:
            y = float(y_np)

        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# ==========================
# Transformer model
# ==========================
class MelTransformer(nn.Module):
    """
    Transformer over the time axis of mel spectrograms.

    Input: (B, C, n_mels, T)
    - Flatten channels and frequency per frame -> feature dim = C * n_mels
    - Optionally subsample time axis
    - Project to d_model
    - Add learnable positional embeddings
    - TransformerEncoder (batch_first)
    - Mean pool over time
    - Linear -> 1 logit
    """
    def __init__(
        self,
        in_channels=3,
        n_mels=64,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        time_subsample=4,
        max_seq_len=256,  # after subsampling
        num_classes=1,
    ):
        super(MelTransformer, self).__init__()

        self.in_channels = in_channels
        self.n_mels = n_mels
        self.d_model = d_model
        self.time_subsample = time_subsample
        self.max_seq_len = max_seq_len

        input_dim = in_channels * n_mels  # e.g. 3 * 64 = 192
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # x: (B, C, n_mels, T)
        B, C, F, T = x.shape

        # (B, T, C, F)
        x = x.permute(0, 3, 1, 2)
        # (B, T, C*F)
        x = x.reshape(B, T, C * F)

        # optional time subsampling to reduce sequence length
        if self.time_subsample > 1:
            x = x[:, ::self.time_subsample, :]  # (B, T', C*F)

        T_sub = x.size(1)
        if T_sub > self.max_seq_len:
            # Truncate if still too long
            x = x[:, :self.max_seq_len, :]
            T_sub = self.max_seq_len

        # Project to d_model
        x = self.input_proj(x)  # (B, T_sub, d_model)

        # Add positional embedding
        pos = self.pos_embed[:, :T_sub, :]
        x = x + pos

        # Transformer encoder
        x = self.encoder(x)  # (B, T_sub, d_model)

        # Mean pool over time
        x = x.mean(dim=1)  # (B, d_model)
        x = self.dropout(x)

        logits = self.classifier(x)  # (B, 1)
        return logits


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


def split_10_patients(paired_files, seed=42):
    """
    Pick 10 random patients (files) and split:
    - 7 train
    - 2 val
    - 1 test
    """
    if len(paired_files) < 10:
        raise ValueError(f"Need at least 10 female files, found {len(paired_files)}.")

    random.Random(seed).shuffle(paired_files)
    subset = paired_files[:10]

    train_pairs = subset[:7]
    val_pairs   = subset[7:9]
    test_pairs  = subset[9:10]

    print(f"[INFO] Using only 10 female patients (files):")
    print(f"       Train files: {len(train_pairs)}")
    print(f"       Val files:   {len(val_pairs)}")
    print(f"       Test files:  {len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs


def fast_compute_pos_weight(dataset):
    """
    Compute pos_weight from full train dataset.
    """
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


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx=1, log_interval=100):
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

        # optional gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * X.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            print(
                f"[Epoch {epoch_idx}] Batch {batch_idx + 1}/{total_batches} "
                f"- Loss: {loss.item():.4f}"
            )

    return running_loss / len(loader.dataset)


def get_scores(model, loader, device):
    """
    Run model over loader and return y_true and y_scores (sigmoid outputs).
    """
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
    return y_true, y_scores


def metrics_at_threshold(y_true, y_scores, threshold):
    preds = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, digits=4)
    cm = confusion_matrix(y_true, preds)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) == 2 else None

    # Balanced accuracy = (sens + spec) / 2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (sens + spec)
    else:
        bal_acc = acc

    return acc, auc, bal_acc, report, cm


def find_best_threshold(y_true, y_scores, thresholds=None):
    """
    Sweep thresholds and pick the one with best balanced accuracy.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9

    best_thr = 0.5
    best_bal_acc = -1.0
    best_stats = None

    for thr in thresholds:
        acc, auc, bal_acc, report, cm = metrics_at_threshold(y_true, y_scores, thr)
        print(f"[THR SWEEP] thr={thr:.2f}  acc={acc:.4f}  bal_acc={bal_acc:.4f}")
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thr = thr
            best_stats = (acc, auc, bal_acc, report, cm)

    print(f"[INFO] Best threshold on val: {best_thr:.3f} (balanced accuracy={best_bal_acc:.4f})")
    return best_thr, best_stats


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

    # 1) Pair all female files, then select 10 patients
    paired_files = find_file_pairs(MEL_DIR, LABEL_DIR)
    train_pairs, val_pairs, test_pairs = split_10_patients(paired_files, seed=RANDOM_SEED)

    # 2) Datasets from those 10 files only
    train_dataset = SleepApneaDataset(train_pairs)
    val_dataset   = SleepApneaDataset(val_pairs)
    test_dataset  = SleepApneaDataset(test_pairs)

    # 3) DataLoaders
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

    # 4) pos_weight from TRAIN dataset
    pos_weight = fast_compute_pos_weight(train_dataset).to(DEVICE)

    # 5) Model, loss, optimizer
    model = MelTransformer(
        in_channels=3,
        n_mels=64,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        time_subsample=TIME_SUBSAMPLE,
        max_seq_len=256,
        num_classes=1,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print("[INFO] Model:")
    print(model)

    # 6) Training loop
    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch_idx=epoch)

        # quick eval at thr=0.5
        y_val_true, y_val_scores = get_scores(model, val_loader, DEVICE)
        acc_05, auc_05, bal_acc_05, report_05, cm_05 = metrics_at_threshold(y_val_true, y_val_scores, 0.5)

        print(f"[VAL] Train Loss: {train_loss:.4f}")
        print(f"[VAL] Accuracy@0.5:     {acc_05:.4f}")
        print(f"[VAL] BalancedAcc@0.5:  {bal_acc_05:.4f}")
        if auc_05 is not None:
            print(f"[VAL] ROC-AUC:         {auc_05:.4f}")
        print("[VAL] Classification report (thr=0.5):")
        print(report_05)

        if auc_05 is not None and auc_05 > best_val_auc:
            best_val_auc = auc_05
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[INFO] New best val AUC: {best_val_auc:.4f} at epoch {epoch}")

        plot_confusion_matrix(
            cm_05,
            class_names=["Normal (0)", "Event (1)"],
            filename=f"confusion_matrix_val_female_transformer_epoch{epoch}.png",
            title=f"Val Confusion Matrix (female, Epoch {epoch}, thr=0.5)",
        )

    # 7) Load best model (by val AUC at thr=0.5)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
        print(f"\n[INFO] Loaded best model with val AUC = {best_val_auc:.4f}")

    # 8) Threshold tuning on validation set
    print("\n[VAL] Threshold tuning on validation set (female, 10 patients)...")
    y_val_true, y_val_scores = get_scores(model, val_loader, DEVICE)
    best_thr, (val_acc_best, val_auc_best, val_bal_acc_best, val_report_best, val_cm_best) = \
        find_best_threshold(y_val_true, y_val_scores)

    print("\n[VAL] Metrics at best threshold (female, 10 patients):")
    print(f"    Best threshold:          {best_thr:.3f}")
    print(f"    Accuracy:                {val_acc_best:.4f}")
    print(f"    Balanced accuracy:       {val_bal_acc_best:.4f}")
    if val_auc_best is not None:
        print(f"    ROC-AUC (unchanged):     {val_auc_best:.4f}")
    print("    Classification report:")
    print(val_report_best)

    plot_confusion_matrix(
        val_cm_best,
        class_names=["Normal (0)", "Event (1)"],
        filename="confusion_matrix_val_female_transformer_best_thr.png",
        title=f"Val Confusion Matrix (female, best thr={best_thr:.2f})",
    )

    # 9) Final test evaluation at thr=0.5 and at best_thr
    print("\n[TEST] Evaluating on test set (female, 10 patients)...")

    y_test_true, y_test_scores = get_scores(model, test_loader, DEVICE)

    # a) Test at threshold = 0.5
    acc_05, auc_05, bal_acc_05, report_05, cm_05 = metrics_at_threshold(y_test_true, y_test_scores, 0.5)
    print("\n[TEST] Metrics at threshold = 0.5 (female, Transformer):")
    print(f"    Accuracy:          {acc_05:.4f}")
    print(f"    Balanced accuracy: {bal_acc_05:.4f}")
    if auc_05 is not None:
        print(f"    ROC-AUC:           {auc_05:.4f}")
    print("    Classification report:")
    print(report_05)

    plot_confusion_matrix(
        cm_05,
        class_names=["Normal (0)", "Event (1)"],
        filename="confusion_matrix_test_female_transformer_thr0.5.png",
        title="Test Confusion Matrix (female, Transformer, thr=0.5)",
    )

    # b) Test at best validation threshold
    acc_best, auc_best, bal_acc_best, report_best, cm_best = metrics_at_threshold(
        y_test_true, y_test_scores, best_thr
    )
    print(f"\n[TEST] Metrics at best val threshold ({best_thr:.3f}) (female, Transformer):")
    print(f"    Accuracy:          {acc_best:.4f}")
    print(f"    Balanced accuracy: {bal_acc_best:.4f}")
    if auc_best is not None:
        print(f"    ROC-AUC:           {auc_best:.4f}")
    print("    Classification report:")
    print(report_best)

    plot_confusion_matrix(
        cm_best,
        class_names=["Normal (0)", "Event (1)"],
        filename=f"confusion_matrix_test_female_transformer_best_thr{best_thr:.2f}.png",
        title=f"Test Confusion Matrix (female, Transformer, best thr={best_thr:.2f})",
    )


if __name__ == "__main__":
    main()

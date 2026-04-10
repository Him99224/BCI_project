"""
main.py - End-to-end pipeline for EEG Seizure Detection.

Steps:
    1. Load BEED dataset
    2. Preprocess (StandardScaler)
    3. Create DataLoaders
    4. Build SeizureNet model
    5. Train with early stopping
    6. Evaluate & generate plots
"""

import config as cfg
from utils.helpers import set_seed, get_device, print_separator
from src.data_loader import load_data, get_dataloaders
from src.preprocess import preprocess
from src.model import SeizureNet
from src.train import train_model
from src.evaluate import evaluate_model, plot_training_curves, plot_confusion_matrix


def main():
    # ── 0. Reproducibility ──
    set_seed(cfg.RANDOM_STATE)
    device = get_device()

    # ── 1. Load data ──
    print_separator("LOADING DATA")
    X, y = load_data()
    print(f"Raw data shape : X={X.shape}, y={y.shape}")
    print(f"Classes         : {cfg.CLASS_NAMES}")
    print(f"Samples/class   : {[int((y == c).sum()) for c in range(cfg.NUM_CLASSES)]}")

    # ── 2. Train/test split ──
    print_separator("SPLITTING DATA")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y,
    )
    print(f"Train : {X_train.shape[0]} samples")
    print(f"Test  : {X_test.shape[0]} samples")

    # ── 3. Preprocess ──
    print_separator("PREPROCESSING")
    X_train, X_test, scaler = preprocess(X_train, X_test)
    print(f"Scaler: {cfg.SCALER_TYPE}")
    print(f"Train mean ≈ {X_train.mean():.4f}, std ≈ {X_train.std():.4f}")

    # ── 4. DataLoaders ──
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    # ── 5. Model ──
    print_separator("MODEL")
    model = SeizureNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\nTotal parameters: {total_params:,}")

    # ── 6. Train ──
    print_separator("TRAINING")
    history = train_model(model, train_loader, test_loader, device=device)

    # ── 7. Evaluate ──
    print_separator("EVALUATION")
    metrics, y_true, y_pred = evaluate_model(model, test_loader, device=device)

    # ── 8. Plots ──
    print_separator("GENERATING PLOTS")
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred)

    print_separator("DONE")
    print(f"Final Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"Final Test F1       : {metrics['f1']:.4f}")
    print(f"Model saved at      : {cfg.MODEL_SAVE_PATH}")
    print(f"Plots saved in      : {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
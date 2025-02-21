import os
import re
import random
import pandas as pd
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import Metric

# ------------------ Constants & Settings ------------------ #
INPUT_DIR = "data/walk_forward_splits_gap"
OUTPUT_RESULTS = "data/walk_forward_transfer_results.csv"
MODEL_SAVE_DIR = "models/walk_forward_transfer"

HIDDEN_SIZE_OPTIONS = [32, 64]
DROPOUT_OPTIONS = [0.1, 0.2]
LEARNING_RATE_OPTIONS = [1e-3, 5e-4]

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 1

BATCH_SIZE = 64
EPOCHS = 10
NUM_SAMPLES_WINDOW0 = 8  # Set to 8 to test all combinations

MIN_TRAIN_SIZE = 5000
MIN_VAL_SIZE = 1000
MIN_TEST_SIZE = 500

# ------------------ Custom BCE Loss as Metric ------------------ #
class BCEWithLogitsMetric(Metric):
    def __init__(self):
        super().__init__(name="bcewithlogits")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")

    def update(self, y_pred, y_true):
        pred_tensor = y_pred.prediction if hasattr(y_pred, "prediction") else y_pred
        self.y_pred.append(pred_tensor.view(-1))
        self.y_true.append(y_true.view(-1).float())

    def compute(self):
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def loss(self, y_pred, y_true):
        pred_tensor = y_pred.prediction if hasattr(y_pred, "prediction") else y_pred
        return F.binary_cross_entropy_with_logits(pred_tensor.view(-1), y_true.view(-1))

    def to_prediction(self, y_pred, **kwargs):
        pred_tensor = y_pred.prediction if hasattr(y_pred, "prediction") else y_pred
        return torch.sigmoid(pred_tensor)

# ------------------ Custom TFT Class ------------------ #
class CustomTFT(pl.LightningModule):
    def __init__(self, dataset, hidden_size, dropout, learning_rate, output_size=1, log_interval=50):
        super().__init__()
        self.save_hyperparameters()  # Save all constructor arguments
        self.tft = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=hidden_size,
            dropout=dropout,
            output_size=output_size,
            loss=BCEWithLogitsMetric(),
            log_interval=log_interval
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.tft(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_true = y[0] if isinstance(y, tuple) else y
        y_hat = self(x)
        loss = self.tft.loss(y_hat, y_true)
        batch_size = y_true.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_true = y[0] if isinstance(y, tuple) else y
        y_hat = self(x)
        loss = self.tft.loss(y_hat, y_true)
        batch_size = y_true.size(0)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint["hyper_parameters"]
        model = cls(**hparams)
        model.load_state_dict(checkpoint["state_dict"])
        return model

# ------------------ Create TimeSeriesDataSet ------------------ #
def create_tsdataset(df):
    if "timestamp" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' and 'label' columns")
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["time_idx"] = df.index
    if "dummy_id" not in df.columns:
        df["dummy_id"] = 0

    known_reals = [
        "open", "high", "low", "close", "volume",
        "ma_5", "ma_15", "ma_30", "ma_60",
        "atr_14", "rsi_14", "adx",
        "hour_sin", "hour_cos", "min_sin", "min_cos",
        "dow_sin", "dow_cos", "market_regime",
    ]
    known_reals = [c for c in known_reals if c in df.columns]

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="label",
        group_ids=["dummy_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[],
        allow_missing_timesteps=False,
        target_normalizer=None,
    )

# ------------------ Classification Metrics ------------------ #
class ClassificationMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, y_pred_logits, y_true):
        probs = torch.sigmoid(y_pred_logits.view(-1))
        preds = (probs >= 0.5).float()
        y_true = y_true.view(-1)
        return {
            "acc": self.accuracy(preds, y_true).item(),
            "precision": self.precision(preds, y_true).item(),
            "recall": self.recall(preds, y_true).item(),
            "f1": self.f1(preds, y_true).item()
        }

# ------------------ Dynamic TP/SL for PnL Simulation ------------------ #
def simulate_pnl(df_rows, probs, tp_factor=1.5, sl_factor=1.0):
    preds = (probs >= 0.5).astype(int)
    if "close" not in df_rows.columns or "atr_14" not in df_rows.columns:
        return {"total_pnl": None, "num_trades": 0, "avg_pnl_per_trade": None}
    closes = df_rows["close"].values
    atrs = df_rows["atr_14"].values
    gains = []
    for i, pred in enumerate(preds):
        if i >= len(closes) - 1:
            gains.append(0.0)
            continue
        if pred == 1:
            atr_now = atrs[i]
            tp_price = closes[i] + atr_now * tp_factor
            sl_price = closes[i] - atr_now * sl_factor
            future_prices = closes[i+1 : min(i+6, len(closes))]
            hit_tp = any(p >= tp_price for p in future_prices)
            hit_sl = any(p <= sl_price for p in future_prices)
            if hit_tp:
                gain = (tp_price - closes[i]) / closes[i]
                gains.append(gain)
            elif hit_sl:
                gain = (sl_price - closes[i]) / closes[i]
                gains.append(gain)
            else:
                gains.append(0.0)
        else:
            gains.append(0.0)
    total_pnl = sum(gains)
    num_trades = sum(preds)
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
    return {"total_pnl": total_pnl, "num_trades": num_trades, "avg_pnl_per_trade": avg_pnl}

# ------------------ Evaluate Model ------------------ #
def evaluate_model(model, df_test):
    if len(df_test) == 0:
        return {"acc": None, "precision": None, "recall": None, "f1": None,
                "total_pnl": None, "num_trades": 0, "avg_pnl_per_trade": None}
    test_dataset = create_tsdataset(df_test)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12)
    metrics_fn = ClassificationMetrics()
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_true = y[0] if isinstance(y, tuple) else y
            y_hat = model(x)
            pred_tensor = y_hat.prediction if hasattr(y_hat, "prediction") else y_hat
            if isinstance(pred_tensor, tuple):
                pred_tensor = pred_tensor[0]  # Extract first tensor if tuple
            all_preds.append(pred_tensor.view(-1))
            all_targets.append(y_true.view(-1).float())
    y_pred_logits = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    cls_stats = metrics_fn(y_pred_logits, y_true)
    probs = torch.sigmoid(y_pred_logits).cpu().numpy()
    df_test_rows = df_test.iloc[-len(probs):].copy()
    pnl_stats = simulate_pnl(df_test_rows, probs)
    return {
        "acc": cls_stats["acc"],
        "precision": cls_stats["precision"],
        "recall": cls_stats["recall"],
        "f1": cls_stats["f1"],
        "total_pnl": pnl_stats["total_pnl"],
        "num_trades": pnl_stats["num_trades"],
        "avg_pnl_per_trade": pnl_stats["avg_pnl_per_trade"],
    }

# ------------------ Hyperparam Search on First Window ------------------ #
def hyperparam_search_first_window(df_train, df_val, window_index, n_samples):
    combos = [
        {"hidden_size": 32, "dropout": 0.1, "learning_rate": 0.0005},
        {"hidden_size": 32, "dropout": 0.1, "learning_rate": 0.001},
        {"hidden_size": 32, "dropout": 0.2, "learning_rate": 0.0005},
        {"hidden_size": 32, "dropout": 0.2, "learning_rate": 0.001},
        {"hidden_size": 64, "dropout": 0.1, "learning_rate": 0.0005},
        {"hidden_size": 64, "dropout": 0.1, "learning_rate": 0.001},
        {"hidden_size": 64, "dropout": 0.2, "learning_rate": 0.0005},
        {"hidden_size": 64, "dropout": 0.2, "learning_rate": 0.001}
    ]
    # Use all combos or limit to n_samples if fewer are desired
    combos = combos[:n_samples] if n_samples < len(combos) else combos
    best_val_loss = float("inf")
    best_model = None
    best_params = None
    train_dataset = create_tsdataset(df_train)
    val_dataset = create_tsdataset(df_val)
    train_loader = train_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=12)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12)
    for i, combo in enumerate(combos):
        print(f"Window {window_index}, Combo {i+1}/{len(combos)}, {combo}")
        tft = CustomTFT(
            dataset=train_dataset,
            hidden_size=combo["hidden_size"],
            dropout=combo["dropout"],
            learning_rate=combo["learning_rate"],
            output_size=1,
            log_interval=50
        )
        logger = TensorBoardLogger(save_dir="logs", name=f"window_{window_index}_combo_{i}")
        ckpt_dir = os.path.join(MODEL_SAVE_DIR, f"window_{window_index}_combo_{i}")
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min"
        )
        es_cb = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            logger=logger,
            callbacks=[ckpt_cb, es_cb],
            accelerator="auto",
            enable_progress_bar=False
        )
        trainer.fit(tft, train_loader, val_loader)
        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        print(f"Completed Combo {i+1}/{len(combos)}, val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = combo
            best_ckpt_path = ckpt_cb.best_model_path
            if best_ckpt_path:
                best_model = CustomTFT.load_from_checkpoint(best_ckpt_path)
    print(f"First window best combo = {best_params}, val_loss={best_val_loss:.4f}")
    return best_model, best_params, best_val_loss

# ------------------ Fine-Tune Model for Subsequent Windows ------------------ #
def fine_tune_model(prev_model, df_train, df_val, hyperparams, window_index):
    train_dataset = create_tsdataset(df_train)
    tft = CustomTFT(
        dataset=train_dataset,
        hidden_size=hyperparams["hidden_size"],
        dropout=hyperparams["dropout"],
        learning_rate=hyperparams["learning_rate"],
        output_size=1,
        log_interval=50
    )
    tft.load_state_dict(prev_model.state_dict())
    for name, param in tft.named_parameters():
        if "decoder" not in name:
            param.requires_grad = False
    val_dataset = create_tsdataset(df_val)
    train_loader = train_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=12)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12)
    logger = TensorBoardLogger(save_dir="logs", name=f"window_{window_index}_fine_tune")
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(MODEL_SAVE_DIR, f"window_{window_index}_fine_tune"),
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=[ckpt_cb, es_cb],
        accelerator="auto",
        enable_progress_bar=False
    )
    trainer.fit(tft, train_loader, val_loader)
    val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
    if isinstance(val_loss, torch.Tensor):
        val_loss = val_loss.item()
    best_ckpt_path = ckpt_cb.best_model_path
    best_model = tft
    if best_ckpt_path:
        best_model = CustomTFT.load_from_checkpoint(best_ckpt_path)
    return best_model, val_loss

# ------------------ Main Walk-Forward Pipeline ------------------ #
def main():
    print("=== Stage 5: Walk-Forward with Transfer Learning + BCEWithLogitsMetric ===")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    results = []
    all_files = os.listdir(INPUT_DIR)
    pattern = re.compile(r"train_(\d+)\.csv")
    train_files = sorted([f for f in all_files if pattern.match(f)], key=lambda x: int(pattern.match(x).group(1)))
    prev_model = None
    prev_hparams = None
    for idx, train_file in enumerate(train_files):
        window_index = int(pattern.match(train_file).group(1))
        val_file = f"val_{window_index}.csv"
        test_file = f"test_{window_index}.csv"
        path_train = os.path.join(INPUT_DIR, train_file)
        path_val = os.path.join(INPUT_DIR, val_file)
        path_test = os.path.join(INPUT_DIR, test_file)
        if not (os.path.exists(path_val) and os.path.exists(path_test)):
            print(f"Window {window_index}: missing val or test file, skipping.")
            continue
        try:
            df_train = pd.read_csv(path_train, parse_dates=["timestamp"])
            df_val = pd.read_csv(path_val, parse_dates=["timestamp"])
            df_test = pd.read_csv(path_test, parse_dates=["timestamp"])
        except Exception as e:
            print(f"Error reading files for window {window_index}: {e}")
            continue
        if len(df_train) < MIN_TRAIN_SIZE or len(df_val) < MIN_VAL_SIZE or len(df_test) < MIN_TEST_SIZE:
            print(f"Skipping window {window_index}: insufficient data.")
            continue
        print(f"\n=== Window {window_index} ===")
        if idx == 0:
            best_model, best_hparams, best_val_loss = hyperparam_search_first_window(
                df_train, df_val, window_index, NUM_SAMPLES_WINDOW0
            )
            if best_model is None:
                print("No best model found for first window, aborting.")
                return
            prev_model = best_model
            prev_hparams = best_hparams
        else:
            best_model, best_val_loss = fine_tune_model(
                prev_model, df_train, df_val, prev_hparams, window_index
            )
            prev_model = best_model
        test_metrics = evaluate_model(best_model, df_test)
        print(f"Window {window_index} => val_loss={best_val_loss:.4f}, test_metrics={test_metrics}")
        row = {
            "window_index": window_index,
            "train_start": df_train["timestamp"].min(),
            "train_end": df_train["timestamp"].max(),
            "val_start": df_val["timestamp"].min(),
            "val_end": df_val["timestamp"].max(),
            "test_start": df_test["timestamp"].min(),
            "test_end": df_test["timestamp"].max(),
            "train_size": len(df_train),
            "val_size": len(df_val),
            "test_size": len(df_test),
            "val_loss": best_val_loss,
            "hidden_size": prev_hparams["hidden_size"],
            "dropout": prev_hparams["dropout"],
            "lr": prev_hparams["learning_rate"],
            "acc": test_metrics["acc"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "total_pnl": test_metrics["total_pnl"],
            "num_trades": test_metrics["num_trades"],
            "avg_pnl_per_trade": test_metrics["avg_pnl_per_trade"],
        }
        results.append(row)
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_RESULTS, index=False)
    print(f"\nWalk-forward transfer learning results saved to {OUTPUT_RESULTS}")
    print(df_results)
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
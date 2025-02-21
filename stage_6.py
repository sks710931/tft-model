import os
import re
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

# Configuration constants
INPUT_DIR = "data/walk_forward_splits_gap"
OUTPUT_RESULTS = "data/walk_forward_transfer_results.csv"
MODEL_SAVE_DIR = "models/walk_forward_transfer"

HIDDEN_SIZE_OPTIONS = [64, 128]
DROPOUT_OPTIONS = [0.1, 0.2]
LEARNING_RATE_OPTIONS = [1e-3, 5e-4]

MAX_ENCODER_LENGTH = 20  # 1 hour
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 256
EPOCHS = 10
NUM_SAMPLES_WINDOW0 = 8

MIN_TRAIN_SIZE = 5000
MIN_VAL_SIZE = 1000
MIN_TEST_SIZE = 500

# Custom metric for cross-entropy loss
class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")

    def update(self, y_pred, y_true):
        # y_pred: [batch_size, num_classes], y_true: [batch_size]
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def compute(self):
        y_pred = torch.cat(self.y_pred)  # [total_samples, num_classes]
        y_true = torch.cat(self.y_true)  # [total_samples]
        return F.cross_entropy(y_pred, y_true)

    def loss(self, y_pred, y_true):
        class_weights = torch.tensor([2.0, 2.0, 0.5], device=y_pred.device)
        return F.cross_entropy(y_pred, y_true, weight=class_weights)

    def to_prediction(self, y_pred, **kwargs):
        return torch.argmax(y_pred, dim=-1)

# Custom TFT model class
class CustomTFT(pl.LightningModule):
    def __init__(self, dataset, hidden_size, dropout, learning_rate, output_size=3, log_interval=50):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])
        self.tft = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=hidden_size,
            dropout=dropout,
            output_size=output_size,
            loss=CrossEntropyMetric(),
            log_interval=log_interval
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.tft(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_true = y[0].squeeze() if isinstance(y, tuple) else y.squeeze()  # [batch_size]
        y_hat = self(x)  # Output object
        prediction = y_hat.prediction.squeeze(1)  # [batch_size, num_classes]
        loss = self.tft.loss(prediction, y_true)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_true = y[0].squeeze() if isinstance(y, tuple) else y.squeeze()
        y_hat = self(x)
        prediction = y_hat.prediction.squeeze(1)
        loss = self.tft.loss(prediction, y_true)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint["hyper_parameters"]
        model = cls(**hparams)
        model.load_state_dict(checkpoint["state_dict"])
        return model

# Function to create TimeSeriesDataSet
def create_tsdataset(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["time_idx"] = df.index
    df["dummy_id"] = 0
    
    known_reals = ["open", "high", "low", "close", "volume", "ma_5", "ma_15", "ma_30", "ma_60", "ma_120",
                   "atr_14", "atr_60", "rsi_14", "adx", "hour_sin", "hour_cos", "min_sin", "min_cos",
                   "dow_sin", "dow_cos", "market_regime", "volatility_spike", "price_diff_60"]
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
        target_normalizer=None
    )

# Classification metrics calculator
class ClassificationMetrics(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, y_pred_logits, y_true):
        preds = torch.argmax(y_pred_logits, dim=-1)
        y_true = y_true.squeeze()
        return {
            "acc": self.accuracy(preds, y_true).item(),
            "precision": self.precision(preds, y_true).item(),
            "recall": self.recall(preds, y_true).item(),
            "f1": self.f1(preds, y_true).item()
        }

# PnL simulation function
def simulate_pnl(df_rows, preds, tp_factor=0.005, sl_factor=0.002, fee=0.001):
    if "close" not in df_rows.columns:
        return {"total_pnl": None, "num_trades": 0, "avg_pnl_per_trade": None}
    closes = df_rows["close"].values
    gains = []
    num_trades = 0
    
    for i, pred in enumerate(preds):
        if i >= len(closes) - 1 or pred == 2:
            gains.append(0.0)
            continue
        entry_price = closes[i]
        if pred == 1:  # Long
            tp_price = entry_price * (1 + tp_factor)
            sl_price = entry_price * (1 - sl_factor)
            future_prices = closes[i+1 : min(i+21, len(closes))]
            hit_tp = any(p >= tp_price for p in future_prices)
            hit_sl = any(p <= sl_price for p in future_prices)
            if hit_tp:
                gain = (tp_price - entry_price) / entry_price - 2 * fee
                gains.append(gain)
            elif hit_sl:
                gain = (sl_price - entry_price) / entry_price - 2 * fee
                gains.append(gain)
            else:
                gains.append(-2 * fee)
            num_trades += 1
        elif pred == 0:  # Short
            tp_price = entry_price * (1 - tp_factor)
            sl_price = entry_price * (1 + sl_factor)
            future_prices = closes[i+1 : min(i+21, len(closes))]
            hit_tp = any(p <= tp_price for p in future_prices)
            hit_sl = any(p >= sl_price for p in future_prices)
            if hit_tp:
                gain = (entry_price - tp_price) / entry_price - 2 * fee
                gains.append(gain)
            elif hit_sl:
                gain = (entry_price - sl_price) / entry_price - 2 * fee
                gains.append(gain)
            else:
                gains.append(-2 * fee)
            num_trades += 1
    
    total_pnl = sum(gains)
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
    return {"total_pnl": total_pnl, "num_trades": num_trades, "avg_pnl_per_trade": avg_pnl}

# Model evaluation function
def evaluate_model(model, df_test):
    if len(df_test) == 0:
        return {"acc": None, "precision": None, "recall": None, "f1": None,
                "total_pnl": None, "num_trades": 0, "avg_pnl_per_trade": None}
    test_dataset = create_tsdataset(df_test)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12)
    metrics_fn = ClassificationMetrics(num_classes=3)
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_true = y[0].squeeze() if isinstance(y, tuple) else y.squeeze()
            y_hat = model(x)
            pred_tensor = y_hat.prediction.squeeze(1)  # [batch_size, num_classes]
            all_preds.append(pred_tensor)
            all_targets.append(y_true)
    y_pred_logits = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    cls_stats = metrics_fn(y_pred_logits, y_true)
    preds = torch.argmax(y_pred_logits, dim=-1).cpu().numpy()
    df_test_rows = df_test.iloc[-len(preds):].copy()
    pnl_stats = simulate_pnl(df_test_rows, preds)
    return {
        "acc": cls_stats["acc"],
        "precision": cls_stats["precision"],
        "recall": cls_stats["recall"],
        "f1": cls_stats["f1"],
        "total_pnl": pnl_stats["total_pnl"],
        "num_trades": pnl_stats["num_trades"],
        "avg_pnl_per_trade": pnl_stats["avg_pnl_per_trade"],
    }

# Hyperparameter search for the first window
def hyperparam_search_first_window(df_train, df_val, window_index, n_samples):
    combos = [
        {"hidden_size": hs, "dropout": dr, "learning_rate": lr}
        for hs in HIDDEN_SIZE_OPTIONS
        for dr in DROPOUT_OPTIONS
        for lr in LEARNING_RATE_OPTIONS
    ]
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
            output_size=3,
            log_interval=50
        )
        logger = TensorBoardLogger(save_dir="logs", name=f"window_{window_index}_combo_{i}")
        ckpt_dir = os.path.join(MODEL_SAVE_DIR, f"window_{window_index}_combo_{i}")
        ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, filename="best-{epoch}-{val_loss:.4f}", monitor="val_loss", save_top_k=1, mode="min")
        es_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            logger=logger,
            callbacks=[ckpt_cb, es_cb],
            accelerator="auto",
            enable_progress_bar=True
        )
        trainer.fit(tft, train_loader, val_loader)
        val_loss = trainer.callback_metrics.get("val_loss", float("inf")).item()
        print(f"Combo {i+1}/{len(combos)}, val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = combo
            best_model = CustomTFT.load_from_checkpoint(ckpt_cb.best_model_path)
    print(f"Best combo = {best_params}, val_loss={best_val_loss:.4f}")
    return best_model, best_params, best_val_loss

# Fine-tuning function for subsequent windows
def fine_tune_model(prev_model, df_train, df_val, hyperparams, window_index):
    train_dataset = create_tsdataset(df_train)
    tft = CustomTFT(
        dataset=train_dataset,
        hidden_size=hyperparams["hidden_size"],
        dropout=hyperparams["dropout"],
        learning_rate=hyperparams["learning_rate"],
        output_size=3,
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
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(MODEL_SAVE_DIR, f"window_{window_index}_fine_tune"), filename="best-{epoch}-{val_loss:.4f}", monitor="val_loss", save_top_k=1, mode="min")
    es_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=[ckpt_cb, es_cb],
        accelerator="auto",
        enable_progress_bar=True
    )
    trainer.fit(tft, train_loader, val_loader)
    val_loss = trainer.callback_metrics.get("val_loss", float("inf")).item()
    best_model = CustomTFT.load_from_checkpoint(ckpt_cb.best_model_path)
    return best_model, val_loss

# Main execution function
def main():
    print("=== Stage 6: Walk-Forward Training for HFT (0.5% TP, 0.2% SL, 1-Hour Horizon) ===")
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
            print(f"Window {window_index}: missing val/test file, skipping.")
            continue
        
        df_train = pd.read_csv(path_train, parse_dates=["timestamp"])
        df_val = pd.read_csv(path_val, parse_dates=["timestamp"])
        df_test = pd.read_csv(path_test, parse_dates=["timestamp"])
        
        if len(df_train) < MIN_TRAIN_SIZE or len(df_val) < MIN_VAL_SIZE or len(df_test) < MIN_TEST_SIZE:
            print(f"Skipping window {window_index}: insufficient data.")
            continue
        
        print(f"\n=== Window {window_index} ===")
        if idx == 0:
            best_model, best_hparams, best_val_loss = hyperparam_search_first_window(df_train, df_val, window_index, NUM_SAMPLES_WINDOW0)
            if best_model is None:
                print("No best model found, aborting.")
                return
            prev_model = best_model
            prev_hparams = best_hparams
        else:
            best_model, best_val_loss = fine_tune_model(prev_model, df_train, df_val, prev_hparams, window_index)
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
    print(f"\nResults saved to {OUTPUT_RESULTS}")
    print(df_results)

if __name__ == "__main__":
    main()
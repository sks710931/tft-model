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
import optuna
import requests
import json
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, "stage_6.log"),
    maxBytes=1_000_000,
    backupCount=5
)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

INPUT_DIR = "data/walk_forward_splits_gap"
OUTPUT_RESULTS = "data/walk_forward_transfer_results_optuna.json"
EPOCH_RESULTS_FILE = "data/epoch_results.json"
WINDOW_RESULTS_FILE = "data/window_results.json"
OPTUNA_TRIALS_FILE = "data/optuna_trials.json"
MODEL_SAVE_DIR = "models/walk_forward_transfer_optuna"
TELEGRAM_BOT_TOKEN = "8179267789:AAFzP6zeQWtPhPfhekTStzqXtW3MIXebPDY"
TELEGRAM_CHAT_ID = "1159940939"

HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX = 32, 128
DROPOUT_MIN, DROPOUT_MAX = 0.15, 0.35
LEARNING_RATE_MIN, LEARNING_RATE_MAX = 1e-4, 1e-2
MAX_ENCODER_LENGTH = 15
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 512
EPOCHS = 10
N_TRIALS_WINDOW0 = 20
MIN_TRAIN_SIZE = 6000
MIN_VAL_SIZE = 2000
MIN_TEST_SIZE = 2000
DYNAMIC_THRESHOLD = 0.0051  # From Stage 3
MIN_PROFIT = 0.002  # Enforce 0.2% profit

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("atr_5", default=[], dist_reduce_fx="cat")

    def update(self, y_pred, y_true, atr_5=None):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        if atr_5 is not None:
            self.atr_5.append(atr_5)

    def compute(self):
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return F.cross_entropy(y_pred, y_true)

    def loss(self, y_pred, y_true, atr_5=None):
        base_weights = torch.tensor([1.0, 1.0, 2.0], device=y_pred.device)  # Stronger "no trade" bias
        if atr_5 is not None and len(self.atr_5) > 0:
            atr_batch = torch.cat(self.atr_5[-len(y_true):])
            volatility_factor = (atr_5 - atr_5.min()) / (atr_5.max() - atr_5.min() + 1e-8)
            weights = base_weights.clone()
            weights[0] *= (1 + volatility_factor.mean())
            weights[1] *= (1 + volatility_factor.mean())
            weights[2] *= (1 - volatility_factor.mean() * 0.3)
            return F.cross_entropy(y_pred, y_true, weight=weights)
        return F.cross_entropy(y_pred, y_true, weight=base_weights)

    def to_prediction(self, y_pred, **kwargs):
        return torch.argmax(y_pred, dim=-1)

class CustomTFT(pl.LightningModule):
    def __init__(self, dataset, hidden_size, dropout, learning_rate, output_size=3, log_interval=20):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset', 'loss', 'logging_metrics'])
        self.dataset = dataset
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
        y_true = y[0].squeeze() if isinstance(y, tuple) else y.squeeze()
        atr_5 = x["encoder_cont"][:, :, self.dataset.reals.index("atr_5")].squeeze(-1)[:, -1]
        y_hat = self(x)
        prediction = y_hat.prediction.squeeze(1)
        loss = self.tft.loss(prediction, y_true, atr_5=atr_5)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        return loss

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        train_loss = self.trainer.callback_metrics.get("train_loss", float("inf")).item()
        val_loss = self.trainer.callback_metrics.get("val_loss", float("inf")).item()
        window_index = getattr(self.trainer, "window_index", "unknown")
        logger.info(f"Window {window_index}, Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        send_telegram_message(f"Window {window_index}, Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        epoch_result = {
            "window_index": window_index,
            "trial_number": getattr(self.trainer, "trial_number", "fine_tune"),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        if os.path.exists(EPOCH_RESULTS_FILE):
            with open(EPOCH_RESULTS_FILE, "r") as f:
                epoch_data = json.load(f)
        else:
            epoch_data = []
        epoch_data.append(epoch_result)
        with open(EPOCH_RESULTS_FILE, "w") as f:
            json.dump(epoch_data, f, indent=4)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_true = y[0].squeeze() if isinstance(y, tuple) else y.squeeze()
        atr_5 = x["encoder_cont"][:, :, self.dataset.reals.index("atr_5")].squeeze(-1)[:, -1]
        y_hat = self(x)
        prediction = y_hat.prediction.squeeze(1)
        loss = self.tft.loss(prediction, y_true, atr_5=atr_5)
        preds = torch.argmax(prediction, dim=-1)
        num_trades = (preds != 2).sum().item()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        self.log("val_trades", num_trades, on_epoch=True, reduce_fx="sum", batch_size=BATCH_SIZE)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        return [optimizer], [scheduler]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, dataset=None, **kwargs):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint["hyper_parameters"]
        if dataset is None:
            raise ValueError("Dataset must be provided to load the checkpoint.")
        model = cls(dataset=dataset, **hparams)
        model.load_state_dict(checkpoint["state_dict"])
        return model

def create_tsdataset(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["time_idx"] = df.index
    df["dummy_id"] = 0
    
    known_reals = [
        "open", "high", "low", "close", "volume",
        "ma_1", "ma_5", "ma_15", "ma_30",
        "atr_5", "atr_14", "rsi_14", "adx",
        "hour_sin", "hour_cos", "min_sin", "min_cos",
        "dow_sin", "dow_cos", "market_regime",
        "volatility_ratio", "price_diff_5"
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
        target_normalizer=None
    )

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

def simulate_pnl(df_rows, preds, fee=0.002):
    if "close" not in df_rows.columns:
        return {"total_pnl": None, "num_trades": 0, "avg_pnl_per_trade": None}
    closes = df_rows["close"].values
    atr_5 = df_rows["atr_5"].values
    gains = []
    num_trades = 0
    
    for i, pred in enumerate(preds):
        if i >= len(closes) - 1 or pred == 2:
            gains.append(0.0)
            continue
        entry_price = closes[i]
        volatility_factor = atr_5[i] / atr_5.mean()
        threshold = max(DYNAMIC_THRESHOLD * max(0.5, min(2.0, volatility_factor)), fee + MIN_PROFIT)
        future_prices = closes[i+1:min(i+16, len(closes))]
        if pred == 1:
            if any((p / entry_price - 1) - fee >= threshold - fee for p in future_prices):
                net_gain = max((p / entry_price - 1) - fee for p in future_prices)
                gains.append(net_gain)
                num_trades += 1
            else:
                gains.append(-fee)
                num_trades += 1
        elif pred == 0:
            if any((1 - p / entry_price) - fee >= threshold - fee for p in future_prices):
                net_gain = max((1 - p / entry_price) - fee for p in future_prices)
                gains.append(net_gain)
                num_trades += 1
            else:
                gains.append(-fee)
                num_trades += 1
    
    total_pnl = sum(gains)
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
    return {"total_pnl": total_pnl, "num_trades": num_trades, "avg_pnl_per_trade": avg_pnl}

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
            pred_tensor = y_hat.prediction.squeeze(1)
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

def objective(trial, df_train, df_val, window_index):
    hidden_size = trial.suggest_int("hidden_size", HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX)
    dropout = trial.suggest_float("dropout", DROPOUT_MIN, DROPOUT_MAX)
    learning_rate = trial.suggest_float("learning_rate", LEARNING_RATE_MIN, LEARNING_RATE_MAX, log=True)
    
    train_dataset = create_tsdataset(df_train)
    val_dataset = create_tsdataset(df_val)
    train_loader = train_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=12)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12)
    
    tft = CustomTFT(
        dataset=train_dataset,
        hidden_size=hidden_size,
        dropout=dropout,
        learning_rate=learning_rate,
        output_size=3,
        log_interval=20
    )
    
    logger_tb = TensorBoardLogger(save_dir="logs", name=f"window_{window_index}_trial_{trial.number}")
    ckpt_dir = os.path.join(MODEL_SAVE_DIR, f"window_{window_index}_trial_{trial.number}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=logger_tb,
        callbacks=[ckpt_cb, es_cb],
        accelerator="auto",
        enable_progress_bar=True
    )
    trainer.window_index = window_index
    trainer.trial_number = trial.number
    
    trainer.fit(tft, train_loader, val_loader)
    best_val_loss = ckpt_cb.best_model_score.item()
    
    logger.info(f"Window 0, Trial {trial.number}: best_val_loss={best_val_loss:.4f} (from checkpoint)")
    send_telegram_message(f"Window 0, Trial {trial.number}: best_val_loss={best_val_loss:.4f} (from checkpoint)")
    
    trial_data = {
        "trial_number": trial.number,
        "params": {"hidden_size": hidden_size, "dropout": dropout, "learning_rate": learning_rate},
        "best_val_loss": best_val_loss,
        "checkpoint_path": ckpt_cb.best_model_path
    }
    if os.path.exists(OPTUNA_TRIALS_FILE):
        with open(OPTUNA_TRIALS_FILE, "r") as f:
            optuna_data = json.load(f)
    else:
        optuna_data = []
    optuna_data.append(trial_data)
    with open(OPTUNA_TRIALS_FILE, "w") as f:
        json.dump(optuna_data, f, indent=4)
    
    return best_val_loss

def hyperparam_search_first_window(df_train, df_val, window_index, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df_train, df_val, window_index), n_trials=n_trials)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_loss = best_trial.value
    
    train_dataset = create_tsdataset(df_train)
    best_ckpt_dir = os.path.join(MODEL_SAVE_DIR, f"window_{window_index}_trial_{best_trial.number}")
    best_ckpt_files = [f for f in os.listdir(best_ckpt_dir) if f.startswith("best-") and f.endswith(".ckpt")]
    if not best_ckpt_files:
        logger.error(f"No checkpoint found in {best_ckpt_dir}")
        raise FileNotFoundError(f"No checkpoint found in {best_ckpt_dir}")
    best_ckpt_path = os.path.join(best_ckpt_dir, best_ckpt_files[0])
    best_model = CustomTFT.load_from_checkpoint(best_ckpt_path, dataset=train_dataset)
    
    logger.info(f"Best params = {best_params}, val_loss={best_val_loss:.4f}")
    with open("best_hyperparameters_window_0.json", "w") as f:
        json.dump({"window_0": best_params, "val_loss": best_val_loss}, f, indent=4)
    send_telegram_message(f"Window 0 best params saved: {best_params}, val_loss={best_val_loss:.4f}")
    
    return best_model, best_params, best_val_loss

def fine_tune_model(prev_model, df_train, df_val, hyperparams, window_index):
    train_dataset = create_tsdataset(df_train)
    tft = CustomTFT(
        dataset=train_dataset,
        hidden_size=hyperparams["hidden_size"],
        dropout=hyperparams["dropout"],
        learning_rate=hyperparams["learning_rate"] * 0.3,
        output_size=3,
        log_interval=20
    )
    tft.load_state_dict(prev_model.state_dict())
    for name, param in tft.named_parameters():
        if "decoder" not in name and "encoder.layers.-1" not in name:
            param.requires_grad = False
    
    val_dataset = create_tsdataset(df_val)
    train_loader = train_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=12)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12)
    
    logger_tb = TensorBoardLogger(save_dir="logs", name=f"window_{window_index}_fine_tune")
    ckpt_dir = os.path.join(MODEL_SAVE_DIR, f"window_{window_index}_fine_tune")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=logger_tb,
        callbacks=[ckpt_cb, es_cb],
        accelerator="auto",
        enable_progress_bar=True
    )
    trainer.window_index = window_index
    trainer.trial_number = "fine_tune"
    
    trainer.fit(tft, train_loader, val_loader)
    best_val_loss = ckpt_cb.best_model_score.item()
    best_model = CustomTFT.load_from_checkpoint(ckpt_cb.best_model_path, dataset=train_dataset)
    return best_model, best_val_loss

def main():
    logger.info("=== Stage 6: Walk-Forward Training with Optuna (Near-HFT, 45-Min Horizon, 200-300 Trades/Day, 0.2% Min Profit) ===")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    results = []
    all_files = os.listdir(INPUT_DIR)
    pattern = re.compile(r"train_(\d+)\.csv")
    train_files = sorted([f for f in all_files if pattern.match(f)], key=lambda x: int(pattern.match(x).group(1)))
    prev_model = None
    prev_hparams = None
    
    # Load existing results and last model
    if os.path.exists(OUTPUT_RESULTS):
        with open(OUTPUT_RESULTS, "r") as f:
            results = json.load(f)
        if results:
            last_window = max([r["window_index"] for r in results])
            last_result = next(r for r in results if r["window_index"] == last_window)
            prev_hparams = {
                "hidden_size": last_result["hidden_size"],
                "dropout": last_result["dropout"],
                "learning_rate": last_result["lr"]
            }
            last_ckpt_dir = os.path.join(MODEL_SAVE_DIR, f"window_{last_window}_fine_tune")
            last_ckpt_files = [f for f in os.listdir(last_ckpt_dir) if f.startswith("best-") and f.endswith(".ckpt")]
            if last_ckpt_files:
                last_ckpt_path = os.path.join(last_ckpt_dir, last_ckpt_files[0])
                df_train = pd.read_csv(os.path.join(INPUT_DIR, f"train_{last_window}.csv"), parse_dates=["timestamp"])
                train_dataset = create_tsdataset(df_train)
                prev_model = CustomTFT.load_from_checkpoint(last_ckpt_path, dataset=train_dataset)
                logger.info(f"Loaded previous model from {last_ckpt_path} for Window {last_window}")
            start_idx = last_window + 1
        else:
            start_idx = 0
    else:
        with open(OUTPUT_RESULTS, "w") as f:
            json.dump([], f)
        start_idx = 0
    
    if not os.path.exists(WINDOW_RESULTS_FILE):
        with open(WINDOW_RESULTS_FILE, "w") as f:
            json.dump([], f)
    
    for idx, train_file in enumerate(train_files[start_idx:], start=start_idx):
        window_index = int(pattern.match(train_file).group(1))
        val_file = f"val_{window_index}.csv"
        test_file = f"test_{window_index}.csv"
        path_train = os.path.join(INPUT_DIR, train_file)
        path_val = os.path.join(INPUT_DIR, val_file)
        path_test = os.path.join(INPUT_DIR, test_file)
        
        if not (os.path.exists(path_val) and os.path.exists(path_test)):
            logger.info(f"Window {window_index}: missing val/test file, skipping.")
            continue
        
        df_train = pd.read_csv(path_train, parse_dates=["timestamp"])
        df_val = pd.read_csv(path_val, parse_dates=["timestamp"])
        df_test = pd.read_csv(path_test, parse_dates=["timestamp"])
        
        if len(df_train) < MIN_TRAIN_SIZE or len(df_val) < MIN_VAL_SIZE or len(df_test) < MIN_TEST_SIZE:
            logger.info(f"Skipping window {window_index}: insufficient data.")
            continue
        
        logger.info(f"\n=== Window {window_index} ===")
        if idx == 0:
            best_model, best_hparams, best_val_loss = hyperparam_search_first_window(df_train, df_val, window_index, N_TRIALS_WINDOW0)
            if best_model is None:
                logger.error("No best model found, aborting.")
                return
            prev_model = best_model
            prev_hparams = best_hparams
        else:
            if prev_model is None or prev_hparams is None:
                logger.error(f"Previous model or hyperparameters not initialized for Window {window_index}. Aborting.")
                return
            best_model, best_val_loss = fine_tune_model(prev_model, df_train, df_val, prev_hparams, window_index)
            prev_model = best_model
        
        test_metrics = evaluate_model(best_model, df_test)
        logger.info(f"Window {window_index} => val_loss={best_val_loss:.4f}, test_metrics={test_metrics}")
        
        msg = (f"Window {window_index} completed: val_loss={best_val_loss:.4f}, "
               f"acc={test_metrics['acc']:.4f}, total_pnl={test_metrics['total_pnl']:.4f}, "
               f"num_trades={test_metrics['num_trades']}")
        logger.info(msg)
        send_telegram_message(msg)
        
        row = {
            "window_index": window_index,
            "train_start": df_train["timestamp"].min().isoformat(),
            "train_end": df_train["timestamp"].max().isoformat(),
            "val_start": df_val["timestamp"].min().isoformat(),
            "val_end": df_val["timestamp"].max().isoformat(),
            "test_start": df_test["timestamp"].min().isoformat(),
            "test_end": df_test["timestamp"].max().isoformat(),
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
        
        with open(OUTPUT_RESULTS, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Updated {OUTPUT_RESULTS} after window {window_index}")
        
        if os.path.exists(WINDOW_RESULTS_FILE):
            with open(WINDOW_RESULTS_FILE, "r") as f:
                window_data = json.load(f)
        else:
            window_data = []
        window_data.append(row)
        with open(WINDOW_RESULTS_FILE, "w") as f:
            json.dump(window_data, f, indent=4)
        logger.info(f"Updated {WINDOW_RESULTS_FILE} after window {window_index}")
    
    logger.info(f"Final results saved to {OUTPUT_RESULTS}")
    msg = f"Training completed: Results saved to {OUTPUT_RESULTS}"
    send_telegram_message(msg)
    
    hyperparams_dict = {f"window_{row['window_index']}": {
        "hidden_size": row["hidden_size"],
        "dropout": row["dropout"],
        "learning_rate": row["lr"],
        "val_loss": row["val_loss"]
    } for row in results}
    with open("best_hyperparameters_all_windows.json", "w") as f:
        json.dump(hyperparams_dict, f, indent=4)
    logger.info("All windows' best hyperparameters saved to best_hyperparameters_all_windows.json")
    send_telegram_message("All windows' best hyperparameters saved to best_hyperparameters_all_windows.json")
    
    logger.info(pd.DataFrame(results).to_string())

if __name__ == "__main__":
    main()
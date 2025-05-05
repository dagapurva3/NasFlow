import gc
import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ------------------- Logging -------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ------------------- CUDA Setup -------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cudnn.benchmark = True
    logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available. Using CPU. Training may be slower.")


# ------------------- Dataset Prep -------------------
def create_label_csv(image_dir, output_dir, train_ratio=0.8):
    rows = []
    class_map = {}
    current_class = 0
    for cls in sorted(os.listdir(image_dir)):
        cls_path = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        class_map[cls] = current_class
        for img_file in os.listdir(cls_path):
            rows.append((os.path.join(cls, img_file), current_class))
        current_class += 1
    df = pd.DataFrame(rows, columns=["filename", "label"])
    train_df, val_df = train_test_split(
        df, train_size=train_ratio, stratify=df["label"], random_state=42
    )
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_labels.csv"), index=False)
    logger.info(f"Saved {len(train_df)} train and {len(val_df)} validation samples.")
    logger.info(f"Class map: {class_map}")


# ------------------- Visualization -------------------
def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8)):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


# ------------------- Dataset Class -------------------
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, csv_path, processor):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 1]
        encodings = self.processor(images=image, return_tensors="pt")
        encodings = {k: v.squeeze() for k, v in encodings.items()}
        encodings["labels"] = torch.tensor(label)
        return encodings


# ------------------- Metric Callback -------------------
class LiveMetricsLogger(TrainerCallback):
    def __init__(self, writer, trial_num):
        self.writer = writer
        self.trial_num = trial_num

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"live/{k}", v, state.global_step)
                    mlflow.log_metric(
                        f"trial{self.trial_num}_{k}", v, step=state.global_step
                    )


# ------------------- Metrics -------------------
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=1)
    labels = torch.tensor(p.label_ids)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# ------------------- Optuna Objective -------------------
def objective(trial):
    model_name = trial.suggest_categorical(
        "model",
        [
            "facebook/deit-tiny-patch16-224",
            "facebook/deit-small-patch16-224",
            "nvidia/mit-b0",
            "microsoft/resnet-18",
        ],
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_train_epochs", 1, 2)

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )

    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(model.device)
        logger.info("Model architecture:")
        logger.info(model)
    except Exception as e:
        logger.warning(f"Could not display model summary: {e}")

    train_dataset = CustomImageDataset(
        "data/images/Images", "dataset/train_labels.csv", processor
    )
    val_dataset = CustomImageDataset(
        "data/images/Images", "dataset/val_labels.csv", processor
    )

    tb_logdir = f"runs/{model_name.replace('/', '_')}_trial_{trial.number}"
    writer = SummaryWriter(tb_logdir)

    args = TrainingArguments(
        output_dir="./results",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=tb_logdir,
        logging_steps=10,
        report_to=["mlflow"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LiveMetricsLogger(writer, trial.number)],
    )

    with mlflow.start_run():
        logger.info(f"Training model: {model_name}")
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_params(trial.params)
        mlflow.log_metrics(metrics)
        for key, val in metrics.items():
            writer.add_scalar(f"metrics/{key}", val, num_epochs)
        writer.close()
        logger.info(f"Finished training with metrics: {metrics}")

    return metrics["eval_accuracy"]


# ------------------- Main -------------------
if __name__ == "__main__":
    NUM_CLASSES = 120
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("nas_image_classification")
    logger.info("Creating CSVs from image folder...")
    create_label_csv("data/images/Images", "dataset")
    logger.info("Starting NAS optimization")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    logger.info("Best trial:")
    logger.info(study.best_trial.params)
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best trial number: {study.best_trial.number}")
    logger.info("Finished all trials.")

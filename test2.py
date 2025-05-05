
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ------------------- Logging -------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ------------------- CUDA Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return class_map


# ------------------- Visualization -------------------
def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8)):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)

    # If class_names is None, use numerical indices
    if class_names is None:
        class_names = list(range(cm.shape[0]))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_learning_curves(
    train_losses, val_losses, train_accs, val_accs, figsize=(15, 6)
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot losses
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot accuracies
    ax2.plot(train_accs, label="Training Accuracy")
    ax2.plot(val_accs, label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


# ------------------- Dataset Class -------------------
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------- Custom CNN Models -------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, num_blocks, num_classes, dropout_rate=0.5):
        super(CustomResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def create_model(model_type, num_classes, dropout_rate):
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes, dropout_rate)
    elif model_type == "deep_cnn":
        return DeepCNN(num_classes, dropout_rate)
    elif model_type == "resnet":
        return CustomResNet([2, 2, 2], num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def early_stopping(val_losses, patience=3, min_delta=0.01):
    if len(val_losses) < patience + 1:
        return False

    # Check if validation loss hasn't improved for 'patience' epochs
    for i in range(patience):
        if val_losses[-1 - i] < val_losses[-2 - i] - min_delta:
            return False
    return True


# ------------------- Training Loop -------------------
def train_model(
    model, train_loader, val_loader, optimizer, criterion, num_epochs, writer, trial_num
):
    model.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log batch-level metrics
            if (i + 1) % 10 == 0:
                batch_loss = running_loss / 10
                batch_acc = 100 * correct / total
                step = epoch * len(train_loader) + i
                writer.add_scalar("Training/Batch Loss", batch_loss, step)
                writer.add_scalar("Training/Batch Accuracy", batch_acc, step)
                mlflow.log_metric(f"trial{trial_num}_batch_loss", batch_loss, step=step)
                mlflow.log_metric(f"trial{trial_num}_batch_acc", batch_acc, step=step)
                running_loss = 0.0
                correct = 0
                total = 0

        # Calculate epoch training statistics
        train_loss, train_acc = evaluate_model(model, train_loader, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Log epoch-level metrics
        writer.add_scalar("Training/Epoch Loss", train_loss, epoch)
        writer.add_scalar("Validation/Epoch Loss", val_loss, epoch)
        writer.add_scalar("Training/Epoch Accuracy", train_acc, epoch)
        writer.add_scalar("Validation/Epoch Accuracy", val_acc, epoch)

        mlflow.log_metric(f"trial{trial_num}_epoch_train_loss", train_loss, step=epoch)
        mlflow.log_metric(f"trial{trial_num}_epoch_val_loss", val_loss, step=epoch)
        mlflow.log_metric(f"trial{trial_num}_epoch_train_acc", train_acc, step=epoch)
        mlflow.log_metric(f"trial{trial_num}_epoch_val_acc", val_acc, step=epoch)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Create and save learning curve plots
    fig = plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
    fig_path = f"./results/learning_curves_trial_{trial_num}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    mlflow.log_artifact(fig_path)

    return val_acc


def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def predict_and_evaluate(model, test_loader, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # If no class names provided, use numerical indices
    if class_names is None:
        class_names = list(range(NUM_CLASSES))

    # Create confusion matrix
    fig = plot_confusion_matrix(all_labels, all_preds, class_names=class_names)
    cm_path = "./results/confusion_matrix.png"
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    fig.savefig(cm_path)
    plt.close(fig)

    return {"accuracy": accuracy, "f1_score": f1, "confusion_matrix_path": cm_path}


# ------------------- Optuna Objective -------------------
def objective(trial):
    # Model hyperparameters
    model_type = trial.suggest_categorical(
        "model_type", ["simple_cnn", "deep_cnn", "resnet"]
    )
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    # Training hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_epochs = trial.suggest_int("num_epochs", 5, 15)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # Data transformation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets and dataloaders
    train_dataset = CustomImageDataset(
        "data/images/Images", "dataset/train_labels.csv", transform
    )
    val_dataset = CustomImageDataset(
        "data/images/Images", "dataset/val_labels.csv", val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize tensorboard writer
    tb_logdir = f"runs/{model_type}_trial_{trial.number}"
    writer = SummaryWriter(tb_logdir)

    # Create model
    model = create_model(model_type, NUM_CLASSES, dropout_rate)
    logger.info(
        f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Log model summary
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    try:
        model.to(device)
        _ = model(dummy_input)  # Forward pass to ensure model works
        logger.info(f"Model architecture: {model}")
    except Exception as e:
        logger.warning(f"Error running model forward pass: {e}")

    # Train and evaluate model
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "optimizer": optimizer_name,
            }
        )

        logger.info(f"Training model: {model_type}")
        val_accuracy = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            num_epochs,
            writer,
            trial.number,
        )

        # Evaluate on validation set
        metrics = predict_and_evaluate(model, val_loader)
        mlflow.log_metrics(
            {"accuracy": metrics["accuracy"], "f1_score": metrics["f1_score"]}
        )
        mlflow.log_artifact(metrics["confusion_matrix_path"])

        # Log model
        model_path = f"./models/model_trial_{trial.number}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        writer.close()
        logger.info(
            f"Finished trial {trial.number} with accuracy: {metrics['accuracy']:.4f}"
        )

    return metrics["accuracy"]


# ------------------- Main -------------------
if __name__ == "__main__":
    NUM_CLASSES = 120  # Update this according to your dataset
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cnn_nas_image_classification")

    logger.info("Creating CSVs from image folder...")
    create_label_csv("data/images/Images", "dataset")

    logger.info("Starting NAS optimization")
    n_jobs = 2  # Number of parallel trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, n_jobs=n_jobs, timeout=600)

    logger.info("Best trial:")
    logger.info(study.best_trial.params)
    logger.info(f"Best accuracy: {study.best_value:.4f}")
    logger.info(f"Best trial number: {study.best_trial.number}")

    # Create visualization of hyperparameter importance
    try:
        param_importances = optuna.importance.get_param_importances(study)
        logger.info("Parameter importances:")
        for param, importance in param_importances.items():
            logger.info(f"  {param}: {importance:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute parameter importances: {e}")

    logger.info("Finished all trials.")

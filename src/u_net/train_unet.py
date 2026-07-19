import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from src.u_net.dataset import SkinCancerDataset
from src.u_net.unet_model import UNet
from src.u_net.loss import BCEDiceLoss

from src.u_net.metrics import evaluate_batch


###########################################
# Cấu hình
###########################################

TRAIN_IMAGE_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1-2_Training_Input\ISIC2018_Task1-2_Training_Input"
TRAIN_MASK_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1_Training_GroundTruth\ISIC2018_Task1_Training_GroundTruth"

VAL_IMAGE_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1-2_Validation_Input\ISIC2018_Task1-2_Validation_Input"
VAL_MASK_DIR = r"D:\Computer Vision Final Project\Src code\data\data_classification\ISIC2018_Task1_Validation_GroundTruth\ISIC2018_Task1_Validation_GroundTruth"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 10
SEED = 42


###########################################
# Train Function
###########################################

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training on {device}")

    ###########################################
    # Dataset
    ###########################################

    # Dataset dùng để train (có augmentation)
    train_dataset = SkinCancerDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, train=True, image_size=IMAGE_SIZE)

    # Dataset dùng để validate (không augmentation)
    val_dataset = SkinCancerDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, train=False, image_size=IMAGE_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    print(f"Train images : {len(train_dataset)}")
    print(f"Validation   : {len(val_dataset)}")

    ###########################################
    # Model
    ###########################################

    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(device)

    ###########################################
    # Loss
    ###########################################

    criterion = BCEDiceLoss()

    ###########################################
    # Optimizer
    ###########################################

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    ###########################################
    # Scheduler
    ###########################################

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    ###########################################
    # AMP
    ###########################################

    scaler = GradScaler("cuda")

    ###########################################
    # History
    ###########################################

    train_loss_history = []
    val_loss_history = []

    dice_history = []

    iou_history = []

    pixel_acc_history = []

    ###########################################
    # Best Model
    ###########################################

    best_model = copy.deepcopy(model.state_dict())

    best_dice = 0.0

    early_stop_counter = 0

    ###########################################
    # Training Loop
    ###########################################

    for epoch in range(EPOCHS):

        print(f"\n========== Epoch [{epoch + 1}/{EPOCHS}] ==========")

        #######################################
        # TRAIN
        #######################################

        model.train()

        running_loss = 0.0

        for step, (images, masks) in enumerate(train_loader):

            images = images.to(device)

            masks = masks.to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):

                outputs = model(images)

                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"  [Train] Step [{step + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        train_loss_history.append(train_loss)

        #######################################
        # VALIDATION
        #######################################

        model.eval()

        val_running_loss = 0.0

        dice_total = 0.0
        iou_total = 0.0
        pixel_acc_total = 0.0

        print("  Evaluating on validation set...")
        with torch.no_grad():

            for step, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)

                with autocast(device_type="cuda"):
                    outputs = model(images)

                    loss = criterion(outputs, masks)

                val_running_loss += loss.item()

                metrics = evaluate_batch(outputs, masks)

                dice_total += metrics["dice"]
                iou_total += metrics["iou"]
                pixel_acc_total += metrics["pixel_acc"]

                if (step + 1) % 50 == 0 or (step + 1) == len(val_loader):
                    print(f"  [Val] Step [{step + 1}/{len(val_loader)}] | Loss: {loss.item():.4f}")

        #######################################
        # Epoch Statistics
        #######################################

        val_loss = val_running_loss / len(val_loader)

        dice = dice_total / len(val_loader)

        iou = iou_total / len(val_loader)

        pixel_acc = pixel_acc_total / len(val_loader)

        val_loss_history.append(val_loss)

        dice_history.append(dice)

        iou_history.append(iou)

        pixel_acc_history.append(pixel_acc)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print("-" * 60)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]")

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Dice Score : {dice:.4f}")

        print(f"IoU Score  : {iou:.4f}")

        print(f"Pixel Acc  : {pixel_acc * 100:.2f}%")

        print(f"Learning Rate : {current_lr:.6f}")

        #######################################
        # Save Best Model
        #######################################

        if dice > best_dice:

            best_dice = dice

            best_model = copy.deepcopy(model.state_dict())

            torch.save(best_model, "best_model.pth")

            print("Best model updated.")

            early_stop_counter = 0

        else:

            early_stop_counter += 1

            print(
                f"No improvement ({early_stop_counter}/{PATIENCE})"
            )

        #######################################
        # Early Stopping
        #######################################

        if early_stop_counter >= PATIENCE:
            print("\nEarly stopping activated.")

            break

    ###########################################
    # Save Last Model
    ###########################################

    torch.save(model.state_dict(), "last_model.pth")

    print("\nTraining Finished.")

    print(f"Best Dice Score : {best_dice:.4f}")

    ###########################################
    # Plot Loss
    ###########################################

    plt.figure(figsize=(8, 5))

    plt.plot(train_loss_history, label="Train Loss")

    plt.plot(val_loss_history, label="Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("loss_curve.png")

    ###########################################
    # Plot Dice
    ###########################################

    plt.figure(figsize=(8, 5))

    plt.plot(dice_history, label="Dice Score")

    plt.xlabel("Epoch")

    plt.ylabel("Dice")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("dice_curve.png")

    ###########################################
    # Plot IoU
    ###########################################

    plt.figure(figsize=(8, 5))

    plt.plot(iou_history, label="IoU")

    plt.xlabel("Epoch")

    plt.ylabel("IoU")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("iou_curve.png")

    ###########################################
    # Plot Pixel Accuracy
    ###########################################

    plt.figure(figsize=(8, 5))

    plt.plot(pixel_acc_history, label="Pixel Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("pixel_accuracy_curve.png")


###########################################
# Main
###########################################

if __name__ == "__main__":
    train()
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
from dataset.contact_dataset import ContactPairsDataset
from model.contact_model import ContactModel
import wandb
import os
from tqdm import tqdm

def contact_pairs_collate_fn(batch):
    pixel_values = []
    all_pairs = []
    timestamps = []

    for item in batch:
        pixel_values.append(item["pixel_values"])
        all_pairs.append(item["pairs"])
        timestamps.append(item["ts"])

    # If all images are the same size, you can stack them into (B, 3, H, W)
    pixel_values = torch.stack(pixel_values, dim=0)

    return {
        "pixel_values": pixel_values,  # shape (B, 3, H, W)
        "pairs": all_pairs,            # list of lists
        "timestamps": timestamps
    }

def evaluate(model, dataloader, device):
    """
    Run a validation loop to compute average loss and accuracy.
    Returns (val_loss, val_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pairs = 0

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating", total=len(dataloader)):
            # Move batch_data to device
            if isinstance(batch_data, dict):
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(device)

            outputs = model(batch_data)  # {"loss": scalar, "logits": (N, 2)}
            loss = outputs["loss"].item()
            logits = outputs["logits"]  # (N, 2) or might be None if no pairs

            total_loss += loss

            if logits is not None:
                # Compute accuracy
                preds = torch.argmax(logits, dim=-1)  # (N,)
                # We need the ground-truth labels. In your model's code, 
                # you likely store them or we can replicate how you do training.
                # For demonstration, assume "contactModel" also kept them or you store them in outputs.
                # If they're not in 'outputs', you'd need to compute them similarly as in training
                # or modify your model's forward to also return labels for easier evaluation.

                # Suppose your model stores them in "outputs['labels']" (N,) for convenience:
                labels = outputs.get("labels", None)
                if labels is not None:
                    correct = (preds == labels).sum().item()
                    total_correct += correct
                    total_pairs += labels.size(0)

    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    val_acc = (total_correct / total_pairs) if total_pairs > 0 else 0.0

    return val_loss, val_acc


def train(args):
    # ---------------------------------------------
    # 1) Initialize W&B (project/run name, etc.)
    # ---------------------------------------------
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )

    # ---------------------------------------------
    # 2) Load a pretrained, frozen DETR
    # ---------------------------------------------
    print(f"Loading DETR from checkpoint: {args.checkpoint_path}")
    detr = DetrForObjectDetection.from_pretrained(args.checkpoint_path)
    detr.eval()  # freeze DETR's parameters

    # ---------------------------------------------
    # 3) Initialize your ContactModel
    # ---------------------------------------------
    contact_model = ContactModel(
        detr=detr,
        hidden_dim=256,
        num_contact_classes=2
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    contact_model.to(device)

    # Optionally track gradients & parameters with wandb
    wandb.watch(contact_model, log="all")

    # ---------------------------------------------
    # 4) Create training dataset & dataloader
    # ---------------------------------------------
    print("Creating training dataset...")
    train_dataset = ContactPairsDataset(
        data_root_dir=args.data_root_dir,
        seq_list_path=args.seq_list_path,
        processor=None,
        transforms=None
    )
    print(f"Train dataset size: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=contact_pairs_collate_fn   # if needed
    )

    # ---------------------------------------------
    # 5) Create validation dataset & dataloader
    # ---------------------------------------------
    val_loader = None
    if args.val_seq_list_path:
        print("Creating validation dataset...")
        val_dataset = ContactPairsDataset(
            data_root_dir=args.data_root_dir,
            seq_list_path=args.val_seq_list_path,
            processor=None,
            transforms=None
        )
        print(f"Val dataset size: {len(val_dataset)}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=contact_pairs_collate_fn
        )

    # ---------------------------------------------
    # 6) Optimizer
    # ---------------------------------------------
    optimizer = optim.Adam(
        [p for p in contact_model.parameters() if p.requires_grad],
        lr=args.lr
    )

    contact_model.train()
    global_step = 0

    # ---------------------------------------------
    # 7) Training Loop
    # ---------------------------------------------
    for epoch in range(args.num_epochs):
        # Train for one epoch
        running_loss = 0.0
        # Wrap your dataloader in tqdm
        for batch_idx, batch_data in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}"
        ):
            # Move batch_data to device
            if isinstance(batch_data, dict):
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(device)

            outputs = contact_model(batch_data)  # {"loss": ..., "logits": ...}
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training step
            step_loss = loss.item()
            running_loss += step_loss
            global_step += 1

            wandb.log({"train_loss": step_loss, "global_step": global_step})

        # End of epoch: log average training loss
        epoch_avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        wandb.log({"epoch_train_loss": epoch_avg_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {epoch_avg_loss:.4f}")

        # ---------------------------------------------
        # 8) Validation Step (if val_loader exists)
        # ---------------------------------------------
        if val_loader is not None:
            val_loss, val_acc = evaluate(contact_model, val_loader, device)
            # Log validation metrics
            wandb.log({
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch + 1
            })
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        contact_model.train()  # re-enable train mode for next epoch

    print("Training completed.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path or identifier for the DETR pretrained weights.")
    parser.add_argument("--data_root_dir", type=str, required=True,
                        help="Root directory of the data.")
    parser.add_argument("--seq_list_path", type=str, required=True,
                        help="Path to a JSON file listing training sequences.")
    parser.add_argument("--val_seq_list_path", type=str, default=None,
                        help="Path to a JSON file listing validation sequences.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--wandb_project", type=str, default="my_contact_project",
                        help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="contact_run",
                        help="W&B run name.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: 'cuda' or 'cpu'.")

    args = parser.parse_args()
    train(args)
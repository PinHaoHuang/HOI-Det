import argparse
import os

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import DetrForObjectDetection
import wandb
from tqdm import tqdm
import datetime

# -------------------
# Your imports
# -------------------
from dataset.contact_dataset import ContactPairsDataset
from model.contact_model import ContactModel, AdvancedContactModel


def contact_pairs_collate_fn(batch):
    pixel_values = []
    all_pairs = []
    timestamps = []

    for item in batch:
        pixel_values.append(item["pixel_values"])
        all_pairs.append(item["pairs"])
        timestamps.append(item["ts"])

    pixel_values = torch.stack(pixel_values, dim=0)
    return {
        "pixel_values": pixel_values,
        "pairs": all_pairs,
        "timestamps": timestamps
    }


def evaluate(model, dataloader, device):
    """
    Run a validation loop to compute average loss and accuracy.
    Returns (val_loss, val_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_pos_correct = 0
    total_pos_pairs = 0
    total_neg_correct = 0
    total_neg_pairs = 0

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating", total=len(dataloader)):
            # Move batch_data to device
            if isinstance(batch_data, dict):
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(device)

            outputs = model(batch_data)  # {"loss": scalar, "logits": (N, 2)}
            loss = outputs["loss"].item()
            logits = outputs["logits"]  # (N, 2)

            total_loss += loss

            if logits is not None:
                preds = torch.argmax(logits, dim=-1)  # (N,)
                labels = outputs.get("labels", None)
                if labels is not None:
                    pos_mask = labels == 1
                    pos_correct = (preds[pos_mask] == labels[pos_mask]).sum().item()
                    total_pos_correct += pos_correct
                    total_pos_pairs += pos_mask.sum()
                    neg_mask = ~pos_mask
                    neg_correct = (preds[neg_mask] == labels[neg_mask]).sum().item()
                    total_neg_correct += neg_correct
                    total_neg_pairs += neg_mask.sum()

    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    # val_acc = (total_correct / total_pairs) if total_pairs > 0 else 0.0
    val_pos_acc = (total_pos_correct / total_pos_pairs) if total_pos_pairs > 0 else 0.0
    val_neg_acc = (total_neg_correct / total_neg_pairs) if total_neg_pairs > 0 else 0.0

    return val_loss, val_pos_acc, val_neg_acc


def train_one_epoch(model, train_loader, optimizer, device, epoch, world_rank, global_step, wandb_active):
    """
    Train model for one epoch.
    Returns the updated `global_step` and the average epoch loss.
    """
    model.train()
    running_loss = 0.0

    total_pos_correct = 0
    total_pos_pairs = 0
    total_neg_correct = 0
    total_neg_pairs = 0

    for batch_idx, batch_data in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"[Rank {world_rank}] Epoch {epoch + 1}"
    ):
        # Move batch_data to device
        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device)

        outputs = model(batch_data)  # {"loss": ..., "logits": ...}
        loss = outputs["loss"]
        logits = outputs['logits']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_loss = loss.item()
        running_loss += step_loss
        global_step += 1

        
        if logits is not None:
            preds = torch.argmax(logits, dim=-1)  # (N,)
            labels = outputs.get("labels", None)
            if labels is not None:
                pos_mask = labels == 1
                pos_correct = (preds[pos_mask] == labels[pos_mask]).sum().item()
                total_pos_correct += pos_correct
                total_pos_pairs += pos_mask.sum()
                neg_mask = ~pos_mask
                neg_correct = (preds[neg_mask] == labels[neg_mask]).sum().item()
                total_neg_correct += neg_correct
                total_neg_pairs += neg_mask.sum()

        pos_acc = (total_pos_correct / total_pos_pairs) if total_pos_pairs > 0 else 0.0
        neg_acc = (total_neg_correct / total_neg_pairs) if total_neg_pairs > 0 else 0.0

        # Only log on rank 0
        if wandb_active:
            wandb.log({"train_loss": step_loss, 
                       "global_step": global_step, 
                       "train_pos_accu": pos_acc,
                       "train_neg_accu": neg_acc})

    epoch_avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    return global_step, epoch_avg_loss


def main_worker(args):
    """
    Each process runs this function.
    """
    # -------------------------
    # 1) Set up distributed
    # -------------------------
    dist.init_process_group(
        backend='nccl',        # or 'gloo' if CPU-based
        init_method='env://'
    )
    world_size = dist.get_world_size()  # total number of processes
    world_rank = dist.get_rank()        # global rank of this process

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Make sure each process only works on the correct GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


    # -------------------------
    # 2) Prepare logging (rank 0 uses W&B)
    # -------------------------
    wandb_active = (world_rank == 0)
    if wandb_active:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # -------------------------
    # 3) Prepare checkpoint directory for MLPs
    # -------------------------
    # Example: /path/to/some_pretrained_checkpoint -> /path/to/contact_checkpoints
    parent_dir = os.path.dirname(args.detr_ckpt_path)  # get parent directory of checkpoint_path
    
    if wandb_active:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_save_dir = os.path.join(parent_dir, f"{timestamp}-contact")
        os.makedirs(ckpt_save_dir, exist_ok=True)


    # -------------------------
    # 4) Load pretrained DETR
    # -------------------------
    if world_rank == 0:
        print(f"[Rank 0] Loading DETR from checkpoint: {args.detr_ckpt_path}")
    detr = DetrForObjectDetection.from_pretrained(args.detr_ckpt_path)
    detr.eval()  # freeze DETR's parameters

    # -------------------------
    # 5) Initialize ContactModel
    # -------------------------
    if args.baseline:
        contact_model = ContactModel(
            detr=detr,
            hidden_dim=256,
            num_contact_classes=2,
            max_pairs=64,
        ).to(device)
    else:

        contact_model = AdvancedContactModel(
            detr=detr,
            d_model=256,
        ).to(device)

    # Wrap with DistributedDataParallel
    contact_model = torch.nn.parallel.DistributedDataParallel(
        contact_model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    # If using W&B on rank 0, watch model
    if wandb_active:
        wandb.watch(contact_model, log="all")

    # -------------------------
    # 6) Create Datasets & Samplers
    # -------------------------
    if world_rank == 0:
        print(f"[Rank 0] Creating training dataset...")

    train_dataset = ContactPairsDataset(
        data_root_dir=args.data_root_dir,
        seq_list_path=args.train_seq_list_path,
        processor=None,
        transforms=None
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=world_rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=contact_pairs_collate_fn
    )

    val_loader = None
    if args.val_seq_list_path:
        if world_rank == 0:
            print("[Rank 0] Creating validation dataset...")
        val_dataset = ContactPairsDataset(
            data_root_dir=args.data_root_dir,
            seq_list_path=args.val_seq_list_path,
            processor=None,
            transforms=None
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=world_rank,
            shuffle=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=contact_pairs_collate_fn
        )

    # -------------------------
    # 7) Setup Optimizer
    # -------------------------
    optimizer = optim.Adam(
        [p for p in contact_model.parameters() if p.requires_grad],
        lr=args.lr
    )

    # Example: step the LR down by factor of 0.1 every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.8
    )

    # -------------------------
    # 8) Training Loop
    # -------------------------
    global_step = 0
    for epoch in range(args.num_epochs):
        # Shuffle shards each epoch
        train_sampler.set_epoch(epoch)

        global_step, epoch_train_loss = train_one_epoch(
            contact_model,
            train_loader,
            optimizer,
            device,
            epoch,
            world_rank,
            global_step,
            wandb_active
        )

        # -------------------------
        # Validation (rank 0 logs)
        # -------------------------
        val_loss, val_pos_acc, val_neg_acc = 0.0, 0.0, 0.0
        if val_loader is not None:
            val_sampler.set_epoch(epoch)
            val_loss, val_pos_acc, val_neg_acc = evaluate(contact_model, val_loader, device)

        # Aggregate validation stats from all ranks
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_pos_acc_tensor = torch.tensor(val_pos_acc, device=device)
        val_neg_acc_tensor = torch.tensor(val_neg_acc, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_pos_acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_neg_acc_tensor, op=dist.ReduceOp.SUM)
        val_loss_avg = val_loss_tensor.item() / world_size
        val_pos_acc_avg = val_pos_acc_tensor.item() / world_size
        val_neg_acc_avg = val_neg_acc_tensor.item() / world_size

        scheduler.step()

        # Log & Print on rank 0
        if wandb_active:
            wandb.log({
                "epoch_train_loss": epoch_train_loss,
                "val_loss": val_loss_avg,
                "val_pos_acc": val_pos_acc_avg,
                "val_neg_acc": val_neg_acc_avg,
                "epoch": epoch + 1
            })
            print(
                f"[Rank 0] Epoch [{epoch+1}/{args.num_epochs}] | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {val_loss_avg:.4f} | "
                f"Val Pos Acc: {val_pos_acc_avg:.4f} | "
                f"Val Neg Acc: {val_neg_acc_avg:.4f} | "
            )

            # -------------------------
            # 9) Save MLPs or entire model state
            # -------------------------
            # Here we save the entire ContactModel's state dict. If you only want
            # the MLP submodule, you could save `contact_model.module.mlp.state_dict()`
            save_path = os.path.join(ckpt_save_dir, f"epoch_{epoch+1}.pth")
            # torch.save(contact_model.module.state_dict(), save_path)
            contact_model.module.save(save_path)
            print(f"[Rank 0] Saved model checkpoint to: {save_path}")

    if wandb_active:
        print("[Rank 0] Training completed.")
        wandb.finish()

    # Clean up the process group
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detr_ckpt_path", type=str, required=True,
                        help="Path or identifier for the DETR pretrained weights.")
    parser.add_argument("--data_root_dir", type=str, required=True,
                        help="Root directory of the data.")
    parser.add_argument("--train_seq_list_path", type=str, required=True,
                        help="Path to a JSON file listing training sequences.")
    parser.add_argument("--val_seq_list_path", type=str, default=None,
                        help="Path to a JSON file listing validation sequences.")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--wandb_project", type=str, default="my_contact_project",
                        help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="contact_run",
                        help="W&B run name.")
    
    return parser.parse_args()


if __name__ == "__main__":
    """
    Usage:
        torchrun --nproc_per_node=<NUM_GPUS> train_distributed.py --checkpoint_path=...
    or
        python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> train_distributed.py --checkpoint_path=...
    """
    args = parse_args()
    main_worker(args)

import os
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import DetrForObjectDetection, DetrImageProcessor
import wandb
from tqdm import tqdm

# Import your dataset class
from dataset.dataset import HOT3dDETRDatasetAggregator
import argparse

def collate_fn(batch):
    """
    Collate function for DETR:
    - Stacks pixel_values into a single tensor of shape (B, 3, H, W).
    - Leaves labels as a list of dicts (one dict per image).
    """
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    pixel_values = torch.stack(pixel_values, dim=0)
    return pixel_values, labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_seq_list", type=str, default='train.json')
    parser.add_argument("--val_seq_list", type=str, default='val.json')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a local pretrained checkpoint or huggingface model id.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    # You can add other hyperparameters, data paths, etc. here
    return parser.parse_args()

def main(args):
    # --------------------------------------------------
    # 1) Initialize distributed training environment
    # --------------------------------------------------
    # Variables set by torchrun:
    #   WORLD_SIZE (total # of processes)
    #   RANK (unique process index within [0..world_size-1])
    #   LOCAL_RANK (this process index on the current node)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Backend "nccl" is standard for multi-GPU
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    # Each process should only use one GPU – tie process rank to device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 2) Create output directory (only on rank 0)
    # --------------------------------------------------
    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"experiments/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None  # other ranks don't save

    # --------------------------------------------------
    # 3) Build Dataset & Distributed Samplers
    # --------------------------------------------------
    # data_root_dir = '/data/HOT3D_dataset'
    data_root_dir = args.data_root

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # aggregator_json_path is a file that contains a list of JSON annotation paths
    train_dataset = HOT3dDETRDatasetAggregator(
        data_root_dir=data_root_dir,
        seq_list_path=args.train_seq_list,
        processor=processor,
        transforms=None
    )
    val_dataset = HOT3dDETRDatasetAggregator(
        data_root_dir=data_root_dir,
        seq_list_path=args.val_seq_list,
        processor=processor,
        transforms=None
    )

    # DistributedSampler ensures each rank sees a unique subset
    train_sampler = DistributedSampler(train_dataset, 
                                       num_replicas=world_size, 
                                       rank=rank,
                                       shuffle=True)
    val_sampler = DistributedSampler(val_dataset,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=2
    )

    # --------------------------------------------------
    # 4) Prepare Model & Optimizer
    # --------------------------------------------------
    label2id = train_dataset.class_mapping
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)

    checkpoint_path = args.checkpoint if args.checkpoint else "facebook/detr-resnet-50"

    model = DetrForObjectDetection.from_pretrained(
        checkpoint_path,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Move to GPU before wrapping with DDP
    model.to(device)

    # Wrap in DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    lr = args.lr
    lr_backbone = args.lr_backbone
    weight_decay = args.weight_decay
    ttl_epoch = args.epochs

    # Separate parameter groups for the backbone vs. rest
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

    # --------------------------------------------------
    # 5) Initialize wandb (rank 0 only)
    # --------------------------------------------------
    if rank == 0:
        wandb.init(project='detr-hot3d', config={
            "epochs": ttl_epoch,
            "learning_rate": lr,
            "batch_size": train_dataloader.batch_size,
        })

    # --------------------------------------------------
    # 6) Training + Validation Loop
    # --------------------------------------------------
    global_step = 0

    for epoch in range(ttl_epoch):
        # Important for shuffling across epochs in DistributedSampler
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # ---------------------------
        # TRAIN
        # ---------------------------
        model.train()
        loss_log_epoch = {}

        for batch_idx, (pixel_values, labels) in enumerate(tqdm(train_dataloader, 
                                                                desc=f"Epoch {epoch} [Train Rank {rank}]")):
            pixel_values = pixel_values.to(device)
            # Move each dict in labels to device
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss_dict = outputs.loss_dict  # sub-losses (loss_ce, loss_bbox, etc.)

            loss.backward()
            optimizer.step()

            # Accumulate local rank's sub-losses
            for k, v in loss_dict.items():
                v_val = v.item()
                if k not in loss_log_epoch:
                    loss_log_epoch[k] = []
                loss_log_epoch[k].append(v_val)

            # (Optional) step-wise logging on rank 0 only:
            if rank == 0:
                for k, v in loss_dict.items():
                    wandb.log({f"train/step/{k}": v.item(), "global_step": global_step})
            global_step += 1

        # After epoch ends, compute *average sub-loss per rank*
        # For a *global average* across all ranks, you'd do an all-reduce:
        #   For each sub-loss key, sum across ranks and divide by (count * world_size).
        # Here's a simpler approach: just rank 0 logs the local average sub-loss:
        if rank == 0:
            for k, v_list in loss_log_epoch.items():
                avg_sub_loss = sum(v_list) / len(v_list)
                wandb.log({f"train/epoch/{k}": avg_sub_loss, "epoch": epoch})
            print(f"[Rank 0] Finished Epoch {epoch} Training.")

        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()
        val_loss_log_epoch = {}
        
        for batch_idx, (pixel_values, labels) in enumerate(tqdm(val_dataloader, 
                                                                desc=f"Epoch {epoch} [Val Rank {rank}]")):
            with torch.no_grad():
                pixel_values = pixel_values.to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss_dict = outputs.loss_dict

            # Accumulate local rank’s sub-losses
            for k, v in loss_dict.items():
                v_val = v.item()
                if k not in val_loss_log_epoch:
                    val_loss_log_epoch[k] = []
                val_loss_log_epoch[k].append(v_val)

        # For a global average, you'd do dist.all_reduce() for each sub-loss.
        # We show local average on rank 0 for simplicity:
        if rank == 0:
            for k, v_list in val_loss_log_epoch.items():
                avg_val_loss = sum(v_list) / len(v_list)
                wandb.log({f"val/epoch/{k}": avg_val_loss, "epoch": epoch})
            print(f"[Rank 0] Finished Epoch {epoch} Validation.")

        # ---------------------------------------
        # (Optional) Save model each epoch (rank 0 only)
        # ---------------------------------------
        if rank == 0:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            # model is wrapped by DDP, so actual model is model.module
            model.module.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)

    # --------------------------------------------------
    # 7) Final Model Save (rank 0)
    # --------------------------------------------------
    if rank == 0:
        final_path = os.path.join(output_dir, f"final_model")
        os.makedirs(final_path, exist_ok=True)
        model.module.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        print(f"Model saved to: {final_path}")
        wandb.finish()

    # --------------------------------------------------
    # 8) Cleanup
    # --------------------------------------------------
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main(args)

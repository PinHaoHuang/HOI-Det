


import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import DetrImageProcessor  # or DetrFeatureExtractor
from dataset.dataset import HOT3dDETRDatasetAggregator
from transformers import DetrForObjectDetection
from tqdm import tqdm
import wandb
import datetime
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def collate_fn(batch):
    """
    Collate function to handle variable-size images for DETR.
    Typically, the processor returns data that can be stacked,
    but you often keep the 'labels' as a list of dicts.
    """
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    pixel_values = torch.stack(pixel_values)  # shape (batch_size, 3, H, W)
    return pixel_values, labels



if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"experiments/{timestamp}"

# 2. Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    # aggregator_json_path -> a file that itself is a list of JSON annotation paths
    seq_list_path = 'train.json'
    # aggregator.json might look like: ["train_part1.json", "train_part2.json", ...]

    data_root_dir = '/data/HOT3D_dataset'

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = HOT3dDETRDatasetAggregator(
        data_root_dir=data_root_dir,
        seq_list_path=seq_list_path,
        processor=processor,
        transforms=None,  # optionally add your own
    )

    val_dataset = HOT3dDETRDatasetAggregator(
        data_root_dir=data_root_dir,
        seq_list_path="val.json",
        processor=processor,
        transforms=None,  # optionally add your own
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )


   
    label2id = train_dataset.class_mapping
    # label2id = {v: k for k, v in id2label.items()}
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id.keys())

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,        # your custom number of classes
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True   # allows weight shapes to differ for the classification head
    )

    

    lr=1e-4
    lr_backbone=1e-5
    weight_decay=1e-4
    device = "cuda:0"
    ttl_epoch=10

    model.train()
    model = model.to(device=device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                weight_decay=weight_decay)
    
    wandb.init(project='detr-hot3d', config={
        "epochs" : ttl_epoch,
        "learning_rate": lr,
        "batch_size": train_dataloader.batch_size,
    })
    
    iterations = 0
    
    for epoch in range(ttl_epoch):
        loss_log = {
        }
        model.train()
        for batch_idx, (pixel_values, labels) in enumerate(tqdm(train_dataloader)):
            # print("pixel_values shape:", pixel_values.shape) 
            # print("labels sample:", labels[0])
            pixel_mask = None
            # pass to your DETR model for training
            # ...
            # break
            # pixel_values = batch["pixel_values"]
            # pixel_mask = batch["pixel_mask"]
            pixel_values = pixel_values.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            optimizer.zero_grad()

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            loss_dict = outputs.loss_dict

            
            for k,v in loss_dict.items():
                if k not in loss_log:
                    loss_log[k] = []
                loss_log[k].append(v.item())
                wandb.log({f"train/step/{k}" : v.item()})

            if (iterations > 0) and (iterations % 500 == 0):
                iteration_dir = os.path.join(output_dir, f"iterations_{iterations}")
                os.makedirs(iteration_dir, exist_ok=True)

                model.save_pretrained(iteration_dir)

            iterations += 1 

        print('Training Loss: ')        
        for k, v in loss_log.items():
            avg = sum(v)/len(v)
            print(k, avg)
            wandb.log({f"train/epoch/{k}" : avg, "epoch" : epoch})

        model.eval()
        loss_log = {}
        
        for batch_idx, (pixel_values, labels) in enumerate(tqdm(val_dataloader)):
            # print("pixel_values shape:", pixel_values.shape) 
            # print("labels sample:", labels[0])
            pixel_mask = None
        
            pixel_values = pixel_values.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

                loss = outputs.loss
                loss_dict = outputs.loss_dict



            for k,v in loss_dict.items():
                if k not in loss_log:
                    loss_log[k] = []
                loss_log[k].append(v.item())  
                

        print('Validation Loss: ')        
        for k, v in loss_log.items():
            avg = sum(v)/len(v)
            print(k, avg)
            wandb.log({f"val/epoch/{k}" : avg, "epoch" : epoch})

    iteration_dir = os.path.join(output_dir, f"iterations_{iterations}")
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir, exist_ok=True)
        model.save_pretrained(output_dir) 
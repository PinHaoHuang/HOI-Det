import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset.contact_dataset import ContactPairsDataset
from model.contact_model import ContactModel, AdvancedContactModel
import argparse
from transformers import DetrForObjectDetection, DetrImageProcessor
import os
import re
from PIL import Image
import numpy as np
from tqdm import tqdm
from eval_util import ContactEvaluator, intersection_ratio

IMAGE_W = 512
IMAGE_H = 512

def pick_checkpoint(contact_ckpt_dir, model_id=0):
    """
    If model_id = 0, or the requested model_id is not found in 'epoch_*.pth', 
    return the latest epoch checkpoint. Otherwise, return the checkpoint matching the model_id.
    """
    if not os.path.isdir(contact_ckpt_dir):
        raise ValueError(f"Directory does not exist: {contact_ckpt_dir}")

    # Gather all checkpoint files of the form "epoch_X.pth"
    ckpt_files = []
    for fname in os.listdir(contact_ckpt_dir):
        match = re.match(r"epoch_(\d+)\.pth", fname)
        if match:
            epoch_num = int(match.group(1))
            ckpt_files.append((epoch_num, fname))

    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in {contact_ckpt_dir}")

    # Sort by epoch number (ascending)
    ckpt_files.sort(key=lambda x: x[0])  # x = (epoch_num, filename)

    # If model_id <= 0, or not found among the epochs, pick the latest
    if model_id <= 0:
        chosen_epoch, chosen_fname = ckpt_files[-1]
    else:
        # Try to find a matching epoch
        found = False
        for ep, fn in ckpt_files:
            if ep == model_id:
                chosen_epoch, chosen_fname = ep, fn
                found = True
                break
        if not found:
            # Not found => pick the latest
            chosen_epoch, chosen_fname = ckpt_files[-1]

    chosen_path = os.path.join(contact_ckpt_dir, chosen_fname)
    return chosen_path

def unnormalize_bbox(norm_box, img_w, img_h):
    """
    Convert a bounding box from normalized [0,1] coords to pixel coords.

    Args:
        norm_box (list/tuple): [cx, cy, w, h], each in [0,1],
                               meaning center x, center y, width, height.
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.

    Returns:
        list: [cx_px, cy_px, w_px, h_px] in pixel coordinates.
    """
    cx, cy, w, h = norm_box
    cx_px = cx * img_w
    cy_px = cy * img_h
    w_px = w * img_w
    h_px = h * img_h
    return [cx_px, cy_px, w_px, h_px]


def contact_pairs_collate_fn(batch):
    pixel_values = []
    all_pairs = []
    timestamps = []
    img_paths = []

    for item in batch:
        pixel_values.append(item["pixel_values"])
        all_pairs.append(item["pairs"])
        timestamps.append(item["ts"])
        img_paths.append(item["img_path"])

    pixel_values = torch.stack(pixel_values, dim=0)
    return {
        "pixel_values": pixel_values,
        "pairs": all_pairs,
        "timestamps": timestamps,
        "img_paths": img_paths,
    }



def pixel_to_normalized(box_xyxy, img_w, img_h):
    """
    Convert a DETR post-process box in pixel coords [x1,y1,x2,y2]
    to normalized [x1/W, y1/H, x2/W, y2/H].
    """
    x1, y1, x2, y2 = box_xyxy
    return [
        x1 / float(img_w),
        y1 / float(img_h),
        x2 / float(img_w),
        y2 / float(img_h),
    ]

def box_area_norm(box):
    """
    box: [x1, y1, x2, y2] in normalized coords
    returns area in [0,1]
    """
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h

def intersection_area_norm(boxA, boxB):
    """
    Intersection area for two boxes in normalized coords.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    return inter_w * inter_h

def baseline_evaluate(contact_model, evaluator, test_dataset, device, 
                      batch_size=1, score_thresh=0.3, top_k=3,
                      intersection_ratio_thresh=0.2):
    
    contact_model.eval()
    
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=contact_pairs_collate_fn
    )

    image_id = 0

    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        # Move any tensor fields to device
        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device)

        with torch.no_grad():
            # DETR forward pass
            outputs = contact_model(batch_data['pixel_values'])
        


        # Post-process: returns a list of dicts (one per image in this batch)
        # Each dict has: {"scores":..., "labels":..., "boxes":...} in pixel coords
        target_sizes = torch.tensor([[IMAGE_H, IMAGE_W]] * batch_data['pixel_values'].shape[0], device=device)
        det_results = test_dataset.processor.post_process(outputs, target_sizes=target_sizes)
        # det_results[i]["scores"] => shape [N]
        # det_results[i]["labels"] => shape [N]
        # det_results[i]["boxes"]  => shape [N,4] (in pixel coords, usually xyxy)

        _bs = len(batch_data['pairs'])  # how many images in this batch

        for i in range(_bs):
            # Ground truth
            gt_pairs = batch_data["pairs"][i] if len(batch_data["pairs"]) > 0 else []

            # (Optional) parse out which are left/right hand from GT, but for evaluation
            # you only need them if you're visualizing. The evaluator only needs `gt_pairs`.
            gt_left_box = None
            gt_right_box = None
            left_contacted_objs = []
            right_contacted_objs = []
            for pair in gt_pairs:
                hand_name = pair["hand_name"]
                obj_box   = pair["obj_box"]      # normalized coords
                label     = pair["label"]        # 1 if contact
                if hand_name == "left_hand":
                    gt_left_box = pair["hand_box"]
                    if label == 1:
                        left_contacted_objs.append(obj_box)
                elif hand_name == "right_hand":
                    gt_right_box = pair["hand_box"]
                    if label == 1:
                        right_contacted_objs.append(obj_box)

            # ----------------------------------------------------------
            # Build predicted pairs from naive baseline
            # ----------------------------------------------------------
            sample_det = det_results[i]  # dict for the i-th image in this batch
            scores = sample_det["scores"]  # shape [N]
            labels = sample_det["labels"]  # shape [N]
            boxes  = sample_det["boxes"]   # shape [N,4] in pixel coords (x1,y1,x2,y2)

            # Convert all to CPU numpy (if needed)
            scores_np = scores.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            boxes_np  = boxes.detach().cpu().numpy()

            # Filter out very low-score detections
            valid_idx = np.where(scores_np >= score_thresh)[0]
            scores_np = scores_np[valid_idx]
            labels_np = labels_np[valid_idx]
            boxes_np  = boxes_np[valid_idx]

            # Map label id -> label name
            # e.g. test_dataset.id2label = {0: "left_hand", 1: "right_hand", 2: "phone", ...}
            id2label  = contact_model.config.id2label

            # Separate out best left_hand, best right_hand, and "objects"
            left_candidates = []
            right_candidates= []
            object_candidates = []

            for idx in range(len(scores_np)):
                lbl_id   = labels_np[idx]
                lbl_name = id2label[lbl_id]
                score    = scores_np[idx]
                box_px   = boxes_np[idx]  # [x1, y1, x2, y2] in pixel coords
                # convert to normalized coords
                box_norm = pixel_to_normalized(box_px, IMAGE_W, IMAGE_H)

                if lbl_name == "left_hand":
                    left_candidates.append((score, box_norm))
                elif lbl_name == "right_hand":
                    right_candidates.append((score, box_norm))
                else:
                    # treat anything else as an object
                    object_candidates.append((lbl_name, score, box_norm))

            # pick best left and right
            best_left  = None
            if len(left_candidates) > 0:
                best_left = max(left_candidates, key=lambda x: x[0])  # pick by score
            best_right = None
            if len(right_candidates) > 0:
                best_right= max(right_candidates, key=lambda x: x[0])

            # Build pred_pairs
            pred_pairs = []

            # Intersection ratio function
            def intersection_ratio(hand_box, obj_box):
                interA = intersection_area_norm(hand_box, obj_box)
                objA   = box_area_norm(obj_box) + 1e-8
                return interA / objA

            # If we have best_left => check each object
            if best_left is not None:
                left_score, left_box = best_left
                for (obj_lbl, obj_score, obj_box) in object_candidates:
                    ratio = intersection_ratio(left_box, obj_box)
                    if ratio >= intersection_ratio_thresh:
                        pred_pairs.append({
                            "hand_name": "left_hand",
                            "hand_box":  left_box,  # normalized
                            "obj_box":   obj_box,   # normalized
                            "score":     ratio      # naive contact confidence
                        })

            # If we have best_right => check each object
            if best_right is not None:
                right_score, right_box = best_right
                for (obj_lbl, obj_score, obj_box) in object_candidates:
                    ratio = intersection_ratio(right_box, obj_box)
                    if ratio >= intersection_ratio_thresh:
                        pred_pairs.append({
                            "hand_name": "right_hand",
                            "hand_box":  right_box,
                            "obj_box":   obj_box,
                            "score":     ratio
                        })

            # -----------------------------------------------
            # Finally, update evaluator
            evaluator.update(image_id, gt_pairs, pred_pairs)
            image_id += 1

    # After processing the entire dataset:
    mAP = evaluator.compute_map()
    recall_k = evaluator.compute_recall_at_k(K=3)

    print(f"mAP={mAP:.4f}, Recall@3={recall_k:.4f}")
     

def main(args):

    device = args.device

    detr = DetrForObjectDetection.from_pretrained(args.checkpoint_path)
    processor = DetrImageProcessor.from_pretrained(args.checkpoint_path)
    detr.eval()  # freeze DETR's parameters
    
    detr.to(device=device)

    test_dataset = ContactPairsDataset(
        data_root_dir=args.data_root_dir,
        seq_list_path=args.seq_list_path,
        processor=processor,
        transforms=None
    )

    evaluator = ContactEvaluator(hand_iou_thresh=0.5, obj_iou_thresh=0.5)

    # evaluate(contact_model, evaluator, test_dataset, device=device, batch_size=8)
    baseline_evaluate(detr, evaluator, test_dataset, device=device, batch_size=8)
    
    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path or identifier for the DETR pretrained weights.")
    parser.add_argument("--data_root_dir", type=str, required=True,
                        help="Root directory of the data.")
    parser.add_argument("--seq_list_path", type=str, required=True,
                        help="Path to a JSON file listing training sequences.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: 'cuda' or 'cpu'.")
    parser.add_argument("--model_id", type=int, default=0)
   
    
    return parser.parse_args()




if __name__ == "__main__":
    """
    Usage:
        torchrun --nproc_per_node=<NUM_GPUS> train_distributed.py --checkpoint_path=...
    or
        python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> train_distributed.py --checkpoint_path=...
    """
    args = parse_args()
    main(args)
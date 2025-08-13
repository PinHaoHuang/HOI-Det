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


def evaluate(contact_model, evaluator, test_dataset, device, batch_size=1, score_thresh=0.5, top_k=3):
    """
    Example inference loop:
      - Iterates over each sample in test_dataset (batch_size=1).
      - For each image, draws:
         1) Ground-truth left/right hands and contacted objects
         2) Predicted left/right hands and top-k predicted contacted objects

    We do a two-step:
      1) forward_test => raw outputs (pred_boxes, scores, labels, left_contact, right_contact)
      2) postprocess_inference => parse best hands, top-k contacts

    Then we visualize results side-by-side.
    """
    contact_model.eval()
    
    
    # A DataLoader with batch_size=1 for simpler visualization
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=contact_pairs_collate_fn
    )

    from tqdm import tqdm

    image_id = 0

    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        # Move any tensor fields to device
        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device)

        # ---------------------------------------------------
        # 2) Model Inference => Raw + Postprocess
        with torch.no_grad():
            # 2a) Raw inference
            raw_outputs = contact_model(batch_data, train=False)
            # raw_outputs is a list of length 1 (since batch_size=1)
            # each item: {
            #    "pred_boxes": (Q,4),
            #    "scores": (Q,),
            #    "labels": (Q,),
            #    "left_contact": (Q,),
            #    "right_contact": (Q,)
            # }

        # 2b) Postprocess => top_k=3, threshold=score_thresh
        #    (You might also define a separate contact_score_thresh, e.g. 0.2.)
        pp_results = contact_model.postprocess_inference(
            batch_raw_outputs=raw_outputs,
            id2label=contact_model.id2label,
            top_k=top_k,
            det_score_thresh=score_thresh,    # e.g. 0.4 for detection
            contact_score_thresh=0.0          # or some threshold for contact
        )

        _bs = len(pp_results)

        for i in range(_bs):
        
            # Ground truth
            gt_pairs = batch_data["pairs"][i] if len(batch_data["pairs"]) > 0 else []

            # ---------------------------------------------------
            # 1) Ground-Truth boxes for left/right & objects
            gt_left_box  = None
            gt_right_box = None
            left_contacted_objs = []
            right_contacted_objs = []

            for pair in gt_pairs:
                hand_name = pair["hand_name"]  # "left_hand" or "right_hand"
                obj_box   = pair["obj_box"]    # [cx, cy, w, h] in normalized coords
                label     = pair["label"]      # 1 if contact, else 0

                if hand_name == "left_hand":
                    gt_left_box = pair["hand_box"]  # [cx,cy,w,h]
                    if label == 1:
                        left_contacted_objs.append(obj_box)
                elif hand_name == "right_hand":
                    gt_right_box = pair["hand_box"]
                    if label == 1:
                        right_contacted_objs.append(obj_box)

            
            # pp_results is also list of length 1
            sample_pp = pp_results[i]
            

            # build list of predicted contact pairs
            pred_pairs = []
            for hand_info in sample_pp["contact_dets"]:
                hand_box = hand_info["hand_box"]  # normalized [x1,y1,x2,y2]
                if hand_box is None:
                    continue
                hand_name = hand_info["hand_name"]
                for cobj in hand_info["contacts"]:
                    obj_box = cobj["obj_box"]      # normalized [x1,y1,x2,y2]
                    c_score = cobj["contact_score"]
                    pred_pairs.append({
                        "hand_name": hand_name,
                        "hand_box":  hand_box,
                        "obj_box":   obj_box,
                        "score":     c_score
                    })

            
            evaluator.update(image_id, gt_pairs, pred_pairs)
            image_id += 1

    # After processing the entire dataset:
    mAP = evaluator.compute_map()
    recall_k = evaluator.compute_recall_at_k(K=3)

    print(f"mAP={mAP:.4f}, Recall@3={recall_k:.4f}")

def baseline_evaluate(contact_model, evaluator, test_dataset, device, batch_size=1, score_thresh=0.3,top_k=3):
    """
    Example inference loop:
      - Iterates over each sample in test_dataset (batch_size=1).
      - For each image, draws:
         1) Ground-truth left/right hands and contacted objects
         2) Predicted left/right hands and top-k predicted contacted objects

    We do a two-step:
      1) forward_test => raw outputs (pred_boxes, scores, labels, left_contact, right_contact)
      2) postprocess_inference => parse best hands, top-k contacts

    Then we visualize results side-by-side.
    """
    contact_model.eval()
    
    
    # A DataLoader with batch_size=1 for simpler visualization
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=contact_pairs_collate_fn
    )

    from tqdm import tqdm

    image_id = 0

    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        # Move any tensor fields to device
        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device)

        # ---------------------------------------------------
        # 2) Model Inference => Raw + Postprocess
        with torch.no_grad():
            # 2a) Raw inference
            raw_outputs = contact_model(batch_data, train=False)
            # raw_outputs is a list of length 1 (since batch_size=1)
            # each item: {
            #    "pred_boxes": (Q,4),
            #    "scores": (Q,),
            #    "labels": (Q,),
            #    "left_contact": (Q,),
            #    "right_contact": (Q,)
            # }

        # 2b) Postprocess => top_k=3, threshold=score_thresh
        #    (You might also define a separate contact_score_thresh, e.g. 0.2.)
        pp_results = contact_model.postprocess_inference(
            batch_raw_outputs=raw_outputs,
            id2label=contact_model.id2label,
            top_k=top_k,
            det_score_thresh=score_thresh,    # e.g. 0.4 for detection
            contact_score_thresh=0.0          # or some threshold for contact
        )

        _bs = len(pp_results)

        for i in range(_bs):
        
            # Ground truth
            gt_pairs = batch_data["pairs"][i] if len(batch_data["pairs"]) > 0 else []

            # ---------------------------------------------------
            # 1) Ground-Truth boxes for left/right & objects
            gt_left_box  = None
            gt_right_box = None
            left_contacted_objs = []
            right_contacted_objs = []

            for pair in gt_pairs:
                hand_name = pair["hand_name"]  # "left_hand" or "right_hand"
                obj_box   = pair["obj_box"]    # [cx, cy, w, h] in normalized coords
                label     = pair["label"]      # 1 if contact, else 0

                if hand_name == "left_hand":
                    gt_left_box = pair["hand_box"]  # [cx,cy,w,h]
                    if label == 1:
                        left_contacted_objs.append(obj_box)
                elif hand_name == "right_hand":
                    gt_right_box = pair["hand_box"]
                    if label == 1:
                        right_contacted_objs.append(obj_box)

            
            # pp_results is also list of length 1
            sample_pp = pp_results[i]
            
            detections = sample_pp['detections']
           
            # gather all left-hand detections
            left_hands = [d for d in detections if d[1] == "left_hand"]
            right_hands= [d for d in detections if d[1] == "right_hand"]

            # pick the best scoring left-hand (if any)
            best_left = None
            if len(left_hands) > 0:
                best_left = max(left_hands, key=lambda x: x[2])  # x[2] is score

            # pick the best scoring right-hand (if any)
            best_right = None
            if len(right_hands) > 0:
                best_right = max(right_hands, key=lambda x: x[2])

            # Step 2: gather object detections (anything not "left_hand" or "right_hand")
            object_dets = [d for d in detections if d[1] not in ["left_hand", "right_hand"]]

            # Step 3: define an intersection ratio threshold
            intersection_ratio_thresh = score_thresh

            pred_pairs = []

            # Step 4a: If we have a "best_left" hand, check all objects
            if best_left is not None:
                _, _, _, left_box = best_left  # (det_id, label_name, score, box)
                for obj_det in object_dets:
                    # obj_det => (det_id, label_name, score, obj_box)
                    obj_box = obj_det[3]
                    ratio = intersection_ratio(left_box, obj_box)
                    if ratio >= intersection_ratio_thresh:
                        pred_pairs.append({
                            "hand_name": "left_hand",
                            "hand_box":  left_box,
                            "obj_box":   obj_box,
                            "score":     ratio  # use intersection ratio as the naive "confidence"
                        })

            # Step 4b: If we have a "best_right" hand, check all objects
            if best_right is not None:
                _, _, _, right_box = best_right
                for obj_det in object_dets:
                    obj_box = obj_det[3]
                    ratio = intersection_ratio(right_box, obj_box)
                    if ratio >= intersection_ratio_thresh:
                        pred_pairs.append({
                            "hand_name": "right_hand",
                            "hand_box":  right_box,
                            "obj_box":   obj_box,
                            "score":     ratio
                        })
            
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
    
    contact_model = AdvancedContactModel(
        detr=detr,
        d_model=256,
    ).to(device)

    parent_dir = os.path.dirname(args.checkpoint_path)  # get parent directory of checkpoint_path
    contact_ckpt_dir = os.path.join(parent_dir, "contact_checkpoints")
    contact_ckpt_path = pick_checkpoint(contact_ckpt_dir, args.model_id)

    checkpoint = torch.load(contact_ckpt_path, map_location=device)
    contact_model.load_state_dict(checkpoint)

    # 3) Move to device, set to eval, etc.
    contact_model.to(device)
    contact_model.eval()

    test_dataset = ContactPairsDataset(
        data_root_dir=args.data_root_dir,
        seq_list_path=args.seq_list_path,
        processor=processor,
        transforms=None
    )

    evaluator = ContactEvaluator(hand_iou_thresh=0.5, obj_iou_thresh=0.5)

    evaluate(contact_model, evaluator, test_dataset, device=device, batch_size=8)
    baseline_evaluate(contact_model, evaluator, test_dataset, device=device, batch_size=8)
    
    
    

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
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def xywh_to_xyxy(box, img_w, img_h):
    """
    Convert [cx, cy, w, h] normalized in [0,1] to [x1, y1, x2, y2] in absolute coords.
    """
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [x1, y1, x2, y2]

def iou(boxA, boxB):
    """
    Compute IoU for two boxes in [x1, y1, x2, y2] format (absolute coords).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    if boxAArea + boxBArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

class ContactEvaluator:
    def __init__(self, hand_iou_thresh=0.5, obj_iou_thresh=0.5):
        """
        hand_iou_thresh: IoU threshold to match predicted hand box with GT hand box
        obj_iou_thresh:  IoU threshold to match predicted object box with GT object box
        """
        self.hand_iou_thresh = hand_iou_thresh
        self.obj_iou_thresh  = obj_iou_thresh
        
        # We'll store all GT contacts and all predictions in lists:
        self.gt_pairs   = []
        self.pred_pairs = []
        
        # Track number of GT contact pairs per image for recall@K
        self.num_gt_per_image = {}  # image_id -> count
    
    def update(self, image_id, gt_pairs, pred_pairs):
        """
        image_id: unique image identifier (e.g. filename or integer)
        
        gt_pairs: list of dictionaries with:
          {
             "hand_name": "left_hand" or "right_hand",
             "hand_box":  [x1,y1,x2,y2] (normalized in [0,1]),
             "obj_box":   [x1,y1,x2,y2] (normalized in [0,1]),
             "label":     1 or 0
          }
          We only consider entries where label=1.
        
        pred_pairs: list of dictionaries with:
          {
            "hand_name": "left_hand" or "right_hand",
            "hand_box":  [x1,y1,x2,y2] (normalized in [0,1]),
            "obj_box":   [x1,y1,x2,y2] (normalized in [0,1]),
            "score":     float (contact confidence)
          }
        """
        # -- Gather GT contact pairs (label=1)
        contact_count = 0
        for g in gt_pairs:
            if g["label"] == 1:
                self.gt_pairs.append({
                    "image_id":  image_id,
                    "hand_name": g["hand_name"],
                    "hand_box":  g["hand_box"],  # already [x1,y1,x2,y2] normalized
                    "obj_box":   g["obj_box"]    # same
                })
                contact_count += 1
        
        self.num_gt_per_image[image_id] = contact_count
        
        # -- Gather predictions
        for p in pred_pairs:
            # We assume p["hand_box"] and p["obj_box"] are also [x1,y1,x2,y2] normalized
            self.pred_pairs.append({
                "image_id":  image_id,
                "hand_name": p["hand_name"],
                "hand_box":  p["hand_box"],
                "obj_box":   p["obj_box"],
                "score":     p["score"]
            })

    def compute_map(self):
        """
        Computes mean Average Precision (mAP) over all images.
        Each (hand_name, hand_box, obj_box) contact is treated as a detection target.
        
        Returns:
          float: mAP across the entire dataset.
        """
        if len(self.gt_pairs) == 0 or len(self.pred_pairs) == 0:
            return 0.0
        
        # 1) Sort predictions by descending score
        self.pred_pairs.sort(key=lambda x: x["score"], reverse=True)
        
        # 2) Mark which GTs are matched
        gt_used = [False] * len(self.gt_pairs)
        
        # We'll store TPs/FPs for each prediction in order
        tps = []
        fps = []
        
        # 3) Iterate preds from highest to lowest score
        print('Computing MAP ....')

        for pred in tqdm(self.pred_pairs):
            matched_gt_index = -1
            best_iou_sum = -1.0
            
            for gt_idx, gt in enumerate(self.gt_pairs):
                if gt_used[gt_idx]:
                    continue
                if gt["image_id"] != pred["image_id"]:
                    continue
                if gt["hand_name"] != pred["hand_name"]:
                    continue
                
                # Check IoU
                iou_hand   = iou(pred["hand_box"], gt["hand_box"])
                iou_object = iou(pred["obj_box"],  gt["obj_box"])
                
                if iou_hand >= self.hand_iou_thresh and iou_object >= self.obj_iou_thresh:
                    # Evaluate "sum of IoUs" if multiple GT could match
                    iou_sum = iou_hand + iou_object
                    if iou_sum > best_iou_sum:
                        best_iou_sum = iou_sum
                        matched_gt_index = gt_idx
            
            if matched_gt_index >= 0:
                tps.append(1)
                fps.append(0)
                gt_used[matched_gt_index] = True
            else:
                tps.append(0)
                fps.append(1)
        
        # 4) Compute precision/recall curve
        tps_cum = torch.cumsum(torch.tensor(tps), dim=0)
        fps_cum = torch.cumsum(torch.tensor(fps), dim=0)
        
        total_gt = len(self.gt_pairs)
        recall_curve    = tps_cum / (total_gt + 1e-8)
        precision_curve = tps_cum / (tps_cum + fps_cum + 1e-8)

        # 5) AP computation
        recall_curve    = recall_curve.numpy()
        precision_curve = precision_curve.numpy()
        
        # Insert (0,1) at start for standard AP formula
        recall_curve    = np.concatenate(([0.0], recall_curve))
        precision_curve = np.concatenate(([1.0], precision_curve))

        # Enforce monotonic precision
        for i in range(1, len(precision_curve)):
            precision_curve[i] = max(precision_curve[i], precision_curve[i-1])

        # Summation over recall increments
        ap = 0.0
        for i in range(1, len(recall_curve)):
            delta = recall_curve[i] - recall_curve[i-1]
            ap += delta * precision_curve[i]
        
        return ap

    def compute_recall_at_k(self, K=3):
        """
        Compute Recall@K across all images. 
        For each image, take top K predictions (by score).
        Count how many GT pairs are matched among those top-K.
        Return average (over images) of matched_gt / total_gt.
        """
        if len(self.gt_pairs) == 0 or len(self.pred_pairs) == 0:
            return 0.0
        
        # Group predictions by image
        preds_by_image = defaultdict(list)
        for p in self.pred_pairs:
            preds_by_image[p["image_id"]].append(p)
        
        # For quick GT lookups
        gt_by_image = defaultdict(list)
        for idx, g in enumerate(self.gt_pairs):
            gt_by_image[g["image_id"]].append(idx)

        image_ids = list(self.num_gt_per_image.keys())
        
        sum_recall = 0.0
        valid_imgs = 0

        print("Computing Recall ...")
        
        for image_id in tqdm(image_ids):
            num_gt = self.num_gt_per_image[image_id]
            if num_gt == 0:
                # no contact pairs in GT => skip or treat as 0.0
                continue
            
            # Sort preds by descending score
            preds_img = preds_by_image[image_id]
            preds_img = sorted(preds_img, key=lambda x: x["score"], reverse=True)
            preds_topk = preds_img[:K]
            
            matched_gt = set()
            for pred in preds_topk:
                best_gt_idx = -1
                best_sum_iou = -1.0
                for gt_idx in gt_by_image[image_id]:
                    if gt_idx in matched_gt:
                        continue
                    gt = self.gt_pairs[gt_idx]
                    
                    if gt["hand_name"] != pred["hand_name"]:
                        continue
                    iou_hand   = iou(pred["hand_box"], gt["hand_box"])
                    iou_object = iou(pred["obj_box"],  gt["obj_box"])
                    if iou_hand >= self.hand_iou_thresh and iou_object >= self.obj_iou_thresh:
                        sum_iou = iou_hand + iou_object
                        if sum_iou > best_sum_iou:
                            best_sum_iou = sum_iou
                            best_gt_idx = gt_idx
                if best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
            
            recall_img = len(matched_gt) / float(num_gt)
            sum_recall += recall_img
            valid_imgs += 1
        
        if valid_imgs == 0:
            return 0.0
        return sum_recall / valid_imgs

def box_area(box):
    """
    box in [x1, y1, x2, y2] normalized coords.
    Area in [0,1].
    """
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h

def intersection_area(boxA, boxB):
    """
    Intersection area of two boxes in normalized coords.
    """
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    return inter_w * inter_h

def intersection_ratio(hand_box, obj_box):
    interA = intersection_area(hand_box, obj_box)
    objA   = box_area(obj_box) + 1e-8  # to avoid division by zero
    return interA / objA
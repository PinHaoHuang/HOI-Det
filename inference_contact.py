import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset.contact_dataset import ContactPairsDataset
from model.contact_model import ContactModel, AdvancedContactModel
import argparse
from transformers import DetrForObjectDetection, DetrImageProcessor
import os
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

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

def get_contact_save_path(img_path):
    """
    Given an image path, e.g.:
       /.../images/rgb/{TS}.png
    return a path:
       /.../images/contact/{TS}.png

    Also create the 'contact' directory if it doesn't exist.
    """
    # 1) Split into directory and file name
    parent_dir = os.path.dirname(img_path)       # e.g. /.../images/rgb
    file_name  = os.path.basename(img_path)      # e.g. {TS}.png

    # 2) Move one level up => /.../images
    images_dir = os.path.dirname(parent_dir)     # e.g. /.../images

    # 3) Create 'contact' directory path => /.../images/contact
    contact_dir = os.path.join(images_dir, "contact")
    os.makedirs(contact_dir, exist_ok=True)

    # 4) Build the final save path => /.../images/contact/{TS}.png
    save_path = os.path.join(contact_dir, file_name)
    return save_path

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

# Helper function to draw a bounding box with a given color and label
def draw_bbox(ax, box, color='blue', label=None, linewidth=2):
    """
    box: [x0, y0, x1, y1] in pixel coordinates
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    # cx, cy, w, h = box
    # x0 = cx - w/2
    # y0 = cy - h/2

    rect = patches.Rectangle((x0, y0), w, h,
                             linewidth=linewidth,
                             edgecolor=color,
                             facecolor='none')
    ax.add_patch(rect)

    if label is not None:
        ax.text(x0, y0 - 5, label, color=color,
                fontsize=10, fontweight='bold')

def inference_and_plot1(contact_model, test_dataset, device, score_thresh=0.4):
    """
    Example inference loop:
      - Iterates over each sample in test_dataset
      - For each image, draws:
         1) Ground-truth left/right hands and contacted objects
         2) Predicted left/right hands and predicted contacted objects
    """
    contact_model.eval()
    
    # We'll assume a DataLoader with batch_size=1 for simpler visualization
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, collate_fn=contact_pairs_collate_fn)

    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device)
        # Move tensors to device
        pixel_values = batch_data["pixel_values"].to(device)
        # We'll assume there's a way to retrieve the original image (e.g. from dataset) 
        # so we can draw on top of it. For demonstration, let's say test_dataset can
        # give us a PIL or NumPy image via an index:
        img_path = batch_data['img_paths'][0]
        save_path = get_contact_save_path(img_path)
        img = Image.open(img_path).convert("RGB")
        image_np = np.array(img)  
        # (Adapt this to however your dataset is structured)

        # Extract ground-truth pairs (list of dicts for a single item if batch_size=1)
        #   e.g. pairs[0] might be the list of pairs for this sample
        gt_pairs = batch_data["pairs"][0] if len(batch_data["pairs"]) > 0 else []

        # 1) ------------------------------------------
        # Plot the Ground-Truth
        # 1a) Identify GT left-hand and right-hand boxes
        gt_left_box  = None
        gt_right_box = None
        # We'll track objects contacted by left and by right
        left_contacted_objs = []
        right_contacted_objs = []

        for pair in gt_pairs:
            hand_name = pair["hand_name"]  # e.g. "left_hand" or "right_hand"
            obj_box   = pair["obj_box"]    # [cx, cy, w, h]
            label     = pair["label"]      # 1 if contact, 0 otherwise
            if hand_name == "left_hand":
                gt_left_box = pair["hand_box"]
                if label == 1:
                    left_contacted_objs.append(obj_box)
            elif hand_name == "right_hand":
                gt_right_box = pair["hand_box"]
                if label == 1:
                    right_contacted_objs.append(obj_box)

        # Determine if any objects are contacted by *both* hands
        # We'll consider them "red" if the same box is in both lists
        # (In practice you might want to check IoU or exact match, but let's keep it simple)
        both_contacted = []
        for obj_l in left_contacted_objs:
            for obj_r in right_contacted_objs:
                # If they happen to be exactly the same box, call it "both"
                if obj_l == obj_r:
                    both_contacted.append(obj_l)

        # 2) ------------------------------------------
        # Run inference
        with torch.no_grad():
            # forward_test returns something like:
            # [
            #   {
            #       "detections": [
            #           (det_id, label_name, score, [cx, cy, w, h]), ...
            #       ],
            #       "contact_pairs": [
            #           (hand_det_id, obj_det_id, contact_score), ...
            #       ]
            #   },
            #   # one dict per image in the batch
            # ]
            pred_results = contact_model(batch_data, train=False, score_thresh=score_thresh)
            # We'll get predictions for the first (and only) sample
            sample_pred = pred_results[0]
            detections  = sample_pred["detections"]
            contact_pairs = sample_pred["contact_pairs"]

        # 2a) Identify the best predicted left-hand / right-hand boxes
        #     from "detections"
        #     Or you can do it just like in forward_test. We'll do a simpler approach:
        pred_left = None
        pred_right = None
        # We'll gather predicted objects in a dictionary keyed by det_id
        pred_objects_dict = {}

        for (det_id, label_name, scr, box_4d) in detections:
            if label_name == "left_hand":
                # For a single image, we might only expect 0-1 left_hand, 
                # but if there's more, pick the highest score
                if (pred_left is None) or (scr > pred_left[1]):
                    pred_left = (box_4d, scr)
            elif label_name == "right_hand":
                if (pred_right is None) or (scr > pred_right[1]):
                    pred_right = (box_4d, scr)
            else:
                # store as an object candidate
                pred_objects_dict[det_id] = (label_name, scr, box_4d)

        # 2b) Check contact_pairs. For each (hand_id, obj_id, contact_score),
        #     see if it exceeds threshold => that means predicted contact
        # We'll track which objects are contacted by left or right
        pred_left_contacted_objs = []
        pred_right_contacted_objs = []

        # We must figure out if "hand_id" is the left or right. But in the example code,
        # the forward_test approach picks the best left/right IDs. Let's cross-reference:
        if (pred_left is not None) or (pred_right is not None):
            # We'll get the IDs from forward_test if it was storing them
            # (hand_det_id, obj_det_id, contact_score)
            # we need to see if hand_det_id corresponds to "best_left" or "best_right"
            # In forward_test, "best_left_id" or "best_right_id" was used, 
            # so let's figure out if the hand_det_id == that.
            # We'll do a quick approach: we find the best left or right hand ID from the detections, 
            # but let's see if we can track that from forward_test. 
            # For brevity, let's do the simpler approach: 
            #   we see which detection ID is "left_hand" with the highest score, etc.
            # Because we already see "pred_left"/"pred_right" is the chosen box, but not the ID. 
            # We'll do a dictionary approach:

            # Build a small dict that maps "left_hand_id" -> det_id, if that was chosen as best
            left_hand_det_id = None
            right_hand_det_id = None

            # We'll do a pass on detections again:
            best_left_scr = -1
            best_right_scr = -1
            for (det_id, label_name, scr, box_4d) in detections:
                if label_name == "left_hand" and scr > best_left_scr:
                    left_hand_det_id = det_id
                    best_left_scr = scr
                elif label_name == "right_hand" and scr > best_right_scr:
                    right_hand_det_id = det_id
                    best_right_scr = scr

            # Now parse predicted contact pairs
            for (hand_id, obj_id, c_score) in contact_pairs:
                if c_score >= score_thresh:
                    # If hand_id matches left_hand_det_id => it's left contact
                    if hand_id == left_hand_det_id:
                        # add object to left contacted list
                        if obj_id in pred_objects_dict:
                            pred_left_contacted_objs.append(pred_objects_dict[obj_id][2])
                    # If hand_id matches right_hand_det_id => it's right contact
                    if hand_id == right_hand_det_id:
                        if obj_id in pred_objects_dict:
                            pred_right_contacted_objs.append(pred_objects_dict[obj_id][2])

        # Determine if any object boxes are in both predicted left & right contact
        pred_both_contacted = []
        for box_l in pred_left_contacted_objs:
            for box_r in pred_right_contacted_objs:
                if box_l == box_r:
                    pred_both_contacted.append(box_l)

        # -------------------------------------------------------------
        # 3) Visualization: We'll do a side-by-side figure
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        ax_gt, ax_pred, ax_det = axes

        # 3a) Show the original image in both subplots
        ax_gt.imshow(image_np)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")

        ax_pred.imshow(image_np)
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")

        ax_det.imshow(image_np)
        ax_det.set_title("Detection")
        ax_det.axis("off")

        # 3b) Draw GT left & right boxes (if they exist)
        

        img_h, img_w = image_np.shape[:2]
        # print(gt_right_box)
        if gt_left_box is not None:
            gt_left_box = unnormalize_bbox(gt_left_box, img_w, img_h)
            draw_bbox(ax_gt, gt_left_box, color='blue', label='GT Left')
        if gt_right_box is not None:
            gt_right_box = unnormalize_bbox(gt_right_box, img_w, img_h)
            draw_bbox(ax_gt, gt_right_box, color='green', label='GT Right')

        # Draw objects that are contact in GT
        # If the same object is in "both_contacted", we color it red
        # otherwise color by which hand is contacting it
        # We may need to check if an object appears in both lists
        for obj_box in left_contacted_objs:
           
            obj_box = unnormalize_bbox(obj_box, img_w, img_h)
            draw_bbox(ax_gt, obj_box, color='blue', label='GT Contact')

        for obj_box in right_contacted_objs:
           
            obj_box = unnormalize_bbox(obj_box, img_w, img_h)
            draw_bbox(ax_gt, obj_box, color='green', label='GT Contact')

        # 3c) Draw predicted left & right
        if pred_left is not None:
            box_4d, scr = pred_left
            box_4d = unnormalize_bbox(box_4d, img_w, img_h)
            draw_bbox(ax_pred, box_4d, color='blue', label=f'Pred Left {scr:.2f}')

        if pred_right is not None:
            box_4d, scr = pred_right
            box_4d = unnormalize_bbox(box_4d, img_w, img_h)
            draw_bbox(ax_pred, box_4d, color='green', label=f'Pred Right {scr:.2f}')

        # Draw predicted contact objects
        # If object is predicted in both => color red
        for obj_box in pred_left_contacted_objs:
           
            obj_box = unnormalize_bbox(obj_box, img_w, img_h)
            draw_bbox(ax_pred, obj_box, color='blue', label='Pred Contact')

        for obj_box in pred_right_contacted_objs:
           
            obj_box = unnormalize_bbox(obj_box, img_w, img_h)
            draw_bbox(ax_pred, obj_box, color='green', label='Pred Contact')

        for (det_id, label_name, scr, box_4d) in detections:
            box_4d = unnormalize_bbox(box_4d, img_w, img_h)
            draw_bbox(ax_det, box_4d, color='blue', label=label_name)

        plt.tight_layout()
        # plt.show()

        # Optionally, if you'd like to save the figure:
        # fig.savefig(f"result_{batch_idx}.png")
        fig.savefig(save_path)
        plt.close(fig)


def inference_and_plot(contact_model, test_dataset, device, score_thresh=0.5, top_k=3):
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
        batch_size=1,
        shuffle=False,
        collate_fn=contact_pairs_collate_fn
    )

    from tqdm import tqdm

    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        # Move any tensor fields to device
        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device)

        # We'll assume each batch_data has: "pixel_values", "pairs", "img_paths", etc.
        img_path = batch_data["img_paths"][0]
        save_path = get_contact_save_path(img_path)

        img = Image.open(img_path).convert("RGB")
        image_np = np.array(img)  
        img_h, img_w = image_np.shape[:2]

        # Ground truth
        gt_pairs = batch_data["pairs"][0] if len(batch_data["pairs"]) > 0 else []

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
        # pp_results is also list of length 1
        sample_pp = pp_results[0]
        # sample_pp => {
        #   "detections": [
        #       (det_id, label_name, score, [x1,y1,x2,y2]), ...
        #   ],
        #   "contact_dets": [
        #       {
        #         "hand_name": "left_hand",
        #         "hand_det_id": <some int or None>,
        #         "hand_box": [x1,y1,x2,y2] or None,
        #         "contacts": [
        #           {
        #             "obj_det_id": <int>,
        #             "obj_label": <str>,
        #             "obj_box": [x1,y1,x2,y2],
        #             "contact_score": <float>
        #           }, ...
        #         ]
        #       },
        #       {
        #         "hand_name": "right_hand",
        #         ...
        #       }
        #   ]
        # }

        # We'll parse:
        det_list    = sample_pp["detections"]
        contact_dets= sample_pp["contact_dets"]

        # Each "contact_dets" has two dicts: one for left_hand, one for right_hand
        left_hand_info  = contact_dets[0]
        right_hand_info = contact_dets[1]

        if False:
            # -------------------------------------------------------------
            # 3) Visualization: We'll do a side-by-side figure with 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(14, 6))
            ax_gt, ax_pred, ax_det = axes

            # 3a) Show the original image
            ax_gt.imshow(image_np)
            ax_gt.set_title("Ground Truth")
            ax_gt.axis("off")

            ax_pred.imshow(image_np)
            ax_pred.set_title("Prediction")
            ax_pred.axis("off")

            ax_det.imshow(image_np)
            ax_det.set_title("Detection")
            ax_det.axis("off")

            # -------------------------------------------------------------
            # 4) Draw Ground Truth
            if gt_left_box is not None:
                box_pix = unnormalize_bbox(gt_left_box, img_w, img_h)
                draw_bbox(ax_gt, box_pix, color='blue', label='GT Left')
            if gt_right_box is not None:
                box_pix = unnormalize_bbox(gt_right_box, img_w, img_h)
                draw_bbox(ax_gt, box_pix, color='green', label='GT Right')

            for obj_box in left_contacted_objs:
                box_pix = unnormalize_bbox(obj_box, img_w, img_h)
                draw_bbox(ax_gt, box_pix, color='blue', label='GT Contact')

            for obj_box in right_contacted_objs:
                box_pix = unnormalize_bbox(obj_box, img_w, img_h)
                draw_bbox(ax_gt, box_pix, color='green', label='GT Contact')

            # -------------------------------------------------------------
            # 5) Draw Postprocessed Predictions (Hand + Contact)
            # left_hand_info => { "hand_box": [x1,y1,x2,y2] or None, ... }
            if left_hand_info["hand_box"] is not None:
                box_pix = left_hand_info["hand_box"]
                box_pix = unnormalize_bbox(box_pix, img_w, img_h)
                draw_bbox(ax_pred, box_pix, color='blue', label=f'Pred Left')

                # Then draw the top-k contact objects
                # for contact_obj in left_hand_info["contacts"]:
                contact_obj = left_hand_info["contacts"][0]
                c_obj_box = contact_obj["obj_box"]
                c_score   = contact_obj["contact_score"]
                if (c_score > 0.5):
                    c_obj_box = unnormalize_bbox(c_obj_box, img_w, img_h)
                    draw_bbox(ax_pred, c_obj_box, color='blue', 
                                label=f'C {c_score:.2f}')

            if right_hand_info["hand_box"] is not None:
                box_pix = right_hand_info["hand_box"]
                box_pix = unnormalize_bbox(box_pix, img_w, img_h)
                draw_bbox(ax_pred, box_pix, color='green', label=f'Pred Right')

                # for contact_obj in right_hand_info["contacts"]:
                contact_obj = right_hand_info["contacts"][0]
                c_obj_box = contact_obj["obj_box"]
                c_score   = contact_obj["contact_score"]
                if (c_score > 0.5):
                    c_obj_box = unnormalize_bbox(c_obj_box, img_w, img_h)
                    draw_bbox(ax_pred, c_obj_box, color='green', 
                                label=f'C {c_score:.2f}')

            # -------------------------------------------------------------
            # 6) Draw All Final Detections (thresholded)
            # sample_pp["detections"] => [ (det_id, label_name, score, [x1,y1,x2,y2]) ]
            for (det_id, label_name, scr, box_xyxy) in det_list:
                box_xyxy = unnormalize_bbox(box_xyxy, img_w, img_h)
                draw_bbox(ax_det, box_xyxy, color='blue', label=f'{label_name}:{scr:.2f}')

            plt.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
        
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        if left_hand_info["hand_box"] is not None:
            

            contact_obj = left_hand_info["contacts"][0]
            c_obj_box = contact_obj["obj_box"]
            c_score   = contact_obj["contact_score"]
            c_obj_name = contact_obj['obj_label']
            if (c_score > 0.7) and (c_obj_name != "left_hand") and (c_obj_name != "right_hand"):
                c_obj_box = unnormalize_bbox(c_obj_box, img_w, img_h)
               
                x_min, y_min, x_max, y_max = c_obj_box
                draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)

                text = f"{c_obj_name}_{c_score:.2f}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Draw background rectangle above the bounding box
                background = [(x_min, y_min - text_height), (x_min + text_width, y_min)]
                draw.rectangle(background, fill="red")
                draw.text((x_min, y_min - text_height), text, fill="white", font=font)

        if right_hand_info["hand_box"] is not None:
           

            contact_obj = right_hand_info["contacts"][0]
            c_obj_box = contact_obj["obj_box"]
            c_score   = contact_obj["contact_score"]
            c_obj_name = contact_obj['obj_label']
            # if (c_score > 0.5):
            if (c_score > 0.7) and (c_obj_name != "left_hand") and (c_obj_name != "right_hand"):
                c_obj_box = unnormalize_bbox(c_obj_box, img_w, img_h)
               
                x_min, y_min, x_max, y_max = c_obj_box
                draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)

                text = f"{c_obj_name}_{c_score:.2f}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Draw background rectangle above the bounding box
                background = [(x_min, y_min - text_height), (x_min + text_width, y_min)]
                draw.rectangle(background, fill="red")
                draw.text((x_min, y_min - text_height), text, fill="white", font=font)
        

        # 8) Create a matplotlib figure for saving
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        # plt.title("D Predictions")
        
        plt.savefig(save_path, bbox_inches="tight")

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

    inference_and_plot(contact_model, test_dataset, device=device, score_thresh=0.5)
    

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
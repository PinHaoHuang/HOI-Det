import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

def compute_iou(box1, box2):
    """
    Computes the IoU (Intersection over Union) of two bounding boxes in 
    normalized [x1, y1, x2, y2] format.

    box1: (x1, y1, x2, y2)
    box2: (x1, y1, x2, y2)
    """
    x1A, y1A, x2A, y2A = box1
    x1B, y1B, x2B, y2B = box2

    # Compute intersection
    interX1 = max(x1A, x1B)
    interY1 = max(y1A, y1B)
    interX2 = min(x2A, x2B)
    interY2 = min(y2A, y2B)

    interW = max(0.0, interX2 - interX1)
    interH = max(0.0, interY2 - interY1)
    interArea = interW * interH

    # Compute union
    box1Area = (x2A - x1A) * (y2A - y1A)
    box2Area = (x2B - x1B) * (y2B - y1B)
    unionArea = box1Area + box2Area - interArea

    if unionArea == 0:
        return 0.0

    return interArea / unionArea

class ContactPairsDataset(Dataset):
    """
    A PyTorch Dataset that:
      - Reads each annotated JSON entry (with 'ts' and 'hand_object_data').
      - Loads the corresponding image.
      - Builds pairs that must involve 'left_hand' or 'right_hand' and another object.
      - Assigns a label = 1 if the object is in contact with the hand, else 0.
      - Stores bounding boxes (normalized) for each pair.
      - Computes IoU between boxes in each pair.
    """
    def __init__(
        self, 
        data_root_dir,
        seq_list_path,
        processor=None, 
        transforms=None
    ):
        """
        Args:
            data_root_dir (str): Root directory containing all sequences.
            seq_list_path (str): Path to a JSON file containing a list of sequence names.
                                 Each sequence has a 'hand_object_contact.json' annotation file.
            processor (callable, optional): A function/transform to process the raw image
                                            into model-ready tensors (e.g., HuggingFace's DetrImageProcessor).
            transforms (callable, optional): Additional transforms applied on each PIL image (augmentation, etc.).
        """
        self.data_root_dir = data_root_dir
        self.seq_list_path = seq_list_path
        self.processor = processor
        self.transforms = transforms

        # 1) Read a JSON file of sequence names (e.g. ["seq_001", "seq_002", ...])
        with open(self.seq_list_path, "r") as f:
            seq_list = json.load(f)

        # 2) Load all annotation data from each sequence
        self.entries = []
        for seq_name in seq_list:
            json_file_path = os.path.join(data_root_dir, seq_name, 'hand_object_contact.json')
            if not os.path.exists(json_file_path):
                continue

            with open(json_file_path, "r") as jf:
                file_data = json.load(jf)  # list of entries, each with "ts" + "hand_object_data"
                for entry in file_data:
                    hand_object_data = entry.get("hand_object_data", [])
                    if len(hand_object_data) == 0:
                        continue  # no data at all, skip

                    # Check whether there's at least one hand
                    # so that we can form at least (hand, object) pairs
                    has_hand = any(
                        obj["object_name"] in ["left_hand", "right_hand"]
                        for obj in hand_object_data
                    )
                    if not has_hand:
                        # If no 'left_hand' or 'right_hand', skip
                        continue

                    # Record path for image loading
                    entry["img_dir"] = os.path.join(data_root_dir, seq_name, 'images', 'rgb')
                    self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
          - "pixel_values": The image as a tensor (if processor/transforms provided).
          - "pairs": A list of dicts, each with:
              {
                "hand_box": (x1, y1, x2, y2),
                "obj_box": (x1, y1, x2, y2),
                "hand_name": str,
                "obj_name": str,
                "label": 0 or 1,
                "iou": float
              }
          - "ts": The timestamp (useful for debugging or referencing).
        """
        entry = self.entries[idx]
        ts = entry["ts"]
        img_dir = entry["img_dir"]

        # 1) Load the image
        img_path = os.path.join(img_dir, f"{ts}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # 2) Apply any custom transforms
        if self.transforms:
            image = self.transforms(image)

        # 3) Apply processor (e.g. for DETR)
        if self.processor is not None:
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)  # shape (3, H, W)
        else:
            # Convert PIL -> Tensor manually
            pixel_values = torch.tensor(
                # shape: (H, W, 3)
                (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
                 .view(image.size[1], image.size[0], 3)
                 .float() / 255.0)
            ).permute(2, 0, 1)  # shape (3, H, W)

        hand_object_data = entry["hand_object_data"]

        # Separate hands from other objects
        hands = {}
        objects = []
        for obj_info in hand_object_data:
            name = obj_info["object_name"]
            bbox = obj_info["bbox"]       # normalized [x1, y1, x2, y2]
            contact = obj_info["contact"] # list of contact names (e.g. ["left_hand"])
            
            if name == "left_hand" or name == "right_hand":
                hands[name] = {
                    "bbox": bbox,
                    "contact": contact
                }
            else:
                objects.append({
                    "name": name,
                    "bbox": bbox,
                    "contact": contact
                })

        # 4) Build (hand, object) pairs
        pairs = []
        for hand_name, hand_data in hands.items():
            hand_box = hand_data["bbox"]
            for obj_data in objects:
                obj_box = obj_data["bbox"]
                # Label = 1 if the object is in contact with this hand, else 0
                contact_label = 1 if hand_name in obj_data["contact"] else 0
                iou_val = compute_iou(hand_box, obj_box)

                pairs.append({
                    "hand_box": hand_box,
                    "obj_box": obj_box,
                    "hand_name": hand_name,
                    "obj_name": obj_data["name"],
                    "label": contact_label,
                    "iou": iou_val
                })

        # 5) Package up
        sample = {
            "pixel_values": pixel_values,  # (3, H, W)
            "pairs": pairs,
            "ts": ts,
            "img_path": img_path,
        }
        return sample

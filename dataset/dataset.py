import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

class HOT3dDETRDatasetAggregator(Dataset):
    def __init__(
        self, 
        data_root_dir,
        seq_list_path,
        processor, 
        transforms=None,
        class_mapping=None
    ):
        """
        Args:
            aggregator_json_path (str): Path to a "list of JSON paths" file.
                                        Each path in that list is a JSON file
                                        with your 'ts' and 'hand_object_data'.
            images_dir (str): Base directory containing images, typically in 
                              subfolder 'rgb/{ts}.png'.
            processor (DetrImageProcessor): A DetrImageProcessor (or DetrFeatureExtractor) 
                                            from Hugging Face for DETR.
            transforms (callable, optional): Optional transforms to apply on each PIL image.
            class_mapping (dict, optional): Maps object_name to integer category_id.
        """
        self.seq_list_path = seq_list_path
        self.data_root_dir = data_root_dir
        self.processor = processor
        self.transforms = transforms

        # If no explicit class mapping is provided, define a simple default.
        # Adjust or extend for your classes:
        if class_mapping is None:
            
            self.class_mapping = {
                                  'None': 0, 'left_hand': 1, 'right_hand': 2, 'aria_small': 3, 'birdhouse_toy': 4, 
                                  'bottle_bbq': 5, 'bottle_mustard': 6, 'bottle_ranch': 7, 'bowl': 8, 'can_parmesan': 9, 
                                  'can_soup': 10, 'can_tomato_sauce': 11, 'carton_milk': 12, 'carton_oj': 13, 'cellphone': 14, 
                                  'coffee_pot': 15, 'dino_toy': 16, 'dumbbell_5lb': 17, 'dvd_remote': 18, 'flask': 19, 
                                  'food_vegetables': 20, 'food_waffles': 21, 'holder_black': 22, 'holder_gray': 23, 
                                  'keyboard': 24, 'mouse': 25, 'mug_patterned': 26, 'mug_white': 27, 'plate_bamboo': 28, 
                                  'potato_masher': 29, 'puzzle_toy': 30, 'spatula_red': 31, 'spoon_wooden': 32, 
                                  'vase': 33, 'whiteboard_eraser': 34, 'whiteboard_marker': 35}
        else:
            self.class_mapping = class_mapping

        # 1) Read the aggregator file, which should contain a list of JSON paths
        with open(self.seq_list_path, "r") as f:
            seq_list = json.load(f)  # This should be a list of JSON file paths

        # 2) Aggregate all data into self.data, filtering out empty entries
        self.data = []
        for seq_name in seq_list:
            json_file_path = os.path.join(data_root_dir, seq_name, 'hand_object_contact.json')
            with open(json_file_path, "r") as jf:
                file_data = json.load(jf)  # each file_data is a list of entries
                for entry in file_data:
                    # Only keep entries where hand_object_data is non-empty
                    if len(entry.get("hand_object_data", [])) > 0:
                        entry['img_dir'] = os.path.join(data_root_dir, seq_name, 'images', 'rgb')
                        self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            pixel_values (Tensor): shape (3, H, W) 
            labels (dict): A DETR-format labels dict with bounding boxes, class labels, etc.
        """
        entry = self.data[idx]
        ts = entry["ts"]
        hand_object_data = entry["hand_object_data"]  # list of objects
        img_dir = entry['img_dir']

        # Load the image (assumes it's in self.images_dir/rgb/<ts>.png)
        img_path = os.path.join(img_dir, f"{ts}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        # Convert bounding boxes and object_name to COCO-like annotations:
        annotations = []
        img_w, img_h = image.size
        for obj in hand_object_data:
            object_name = obj["object_name"]
            bbox_norm = obj["bbox"]  # [x_min_norm, y_min_norm, x_max_norm, y_max_norm]

            # Convert normalized to absolute pixel coordinates
            x_min = bbox_norm[0] * img_w
            y_min = bbox_norm[1] * img_h
            x_max = bbox_norm[2] * img_w
            y_max = bbox_norm[3] * img_h

            # [x, y, width, height] for COCO / DETR
            w = x_max - x_min
            h = y_max - y_min

            # Convert object_name to category_id
            category_id = self.class_mapping.get(object_name, 0)  # default 0 if not found

            annotations.append({
                "bbox": [x_min, y_min, w, h],
                "category_id": category_id,
                "area" : w*h
            })

        # Build a COCO-style target
        target = {
            "image_id": idx,  # you can also use ts, but ensure uniqueness in a large dataset
            "annotations": annotations
        }

        # Encode with the DETR processor
        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        # The processor returns batches by default -> remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels

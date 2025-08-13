#!/usr/bin/env python

import argparse
import os
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the DETR checkpoint directory or a Hugging Face model ID.")
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--score_threshold", type=float, default=0.7,
                        help="Confidence threshold for displaying bounding boxes.")
    args = parser.parse_args()

    # 1) Load the model and processor
    print(f"Loading model from: {args.checkpoint_path}")
    model = DetrForObjectDetection.from_pretrained(args.checkpoint_path)
    processor = DetrImageProcessor.from_pretrained(args.checkpoint_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Load the image
    image = Image.open(args.img_path).convert("RGB")
    width, height = image.size

    # 3) Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 4) Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 5) Convert outputs (logits + boxes) to COCO format with processor’s post_process
    target_sizes = torch.tensor([[height, width]], device=device)
    results = processor.post_process(outputs, target_sizes=target_sizes)[0]

    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    boxes = results["boxes"].cpu().numpy()  # shape (num_boxes, 4)

    # 6) Filter boxes by score_threshold
    threshold = args.score_threshold
    selected_indices = [i for i, s in enumerate(scores) if s >= threshold]
    filtered_scores = scores[selected_indices]
    filtered_labels = labels[selected_indices]
    filtered_boxes = boxes[selected_indices]

    # 7) Draw bounding boxes on the original PIL image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
        box = list(map(int, box))  # Convert float coords to int
        x_min, y_min, x_max, y_max = box
        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)

        text = f"{model.config.id2label[label]}: {score:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle above the bounding box
        background = [(x_min, y_min - text_height), (x_min + text_width, y_min)]
        draw.rectangle(background, fill="red")
        draw.text((x_min, y_min - text_height), text, fill="white", font=font)
       

    # 8) Create a matplotlib figure for saving
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title("DETR Predictions")

    # 9) Save to "output/" directory
    os.makedirs("output", exist_ok=True)
    img_basename = os.path.basename(args.img_path)
    out_path = os.path.join("output", f"{os.path.splitext(img_basename)[0]}_pred.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Annotated image saved to: {out_path}")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
from torch.nn import TransformerDecoder, TransformerDecoderLayer


def box_cxcywh_to_xyxy(box):
    """
    Utility to convert [cx, cy, w, h] -> [x1, y1, x2, y2], all normalized to [0, 1].
    """
    cx, cy, w, h = box
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return (x1, y1, x2, y2)


def box_xyxy_to_cxcywh(box):
    """
    Convert [x_min, y_min, x_max, y_max]
    to [cx, cy, w, h].
    """
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    return [cx, cy, w, h]

def cxcywh_to_xyxy_tensor(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [cx, cy, w, h] format to [x1, y1, x2, y2].

    Args:
        boxes (torch.Tensor): Shape (..., 4). The last dimension is 4:
            boxes[..., 0] = cx
            boxes[..., 1] = cy
            boxes[..., 2] = w
            boxes[..., 3] = h

    Returns:
        torch.Tensor: Same shape (..., 4), but in [x1, y1, x2, y2] format.
    """
    # Ensure the input has at least 2 dimensions and last dimension size is 4
    assert boxes.shape[-1] == 4, f"boxes must have last dimension = 4, got {boxes.shape}"

    # Separate the components
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w  = boxes[..., 2]
    h  = boxes[..., 3]

    # Compute xyxy
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)

    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return xyxy

def compute_iou(box_pred, box_gt):
    """
    Compute IoU between two normalized boxes in [cx, cy, w, h] format.
    """
    # Convert both to x1y1x2y2
    x1p, y1p, x2p, y2p = box_cxcywh_to_xyxy(box_pred)
    x1g, y1g, x2g, y2g = box_cxcywh_to_xyxy(box_gt)

    # Intersection
    ix1 = max(x1p, x1g)
    iy1 = max(y1p, y1g)
    ix2 = min(x2p, x2g)
    iy2 = min(y2p, y2g)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    # Union
    pred_area = (x2p - x1p) * (y2p - y1p)
    gt_area   = (x2g - x1g) * (y2g - y1g)
    union_area = pred_area + gt_area - inter_area + 1e-7  # tiny epsilon to avoid div by zero

    iou_val = inter_area / union_area
    return iou_val

def compute_iou_batch(pred_boxes, gt_box):
    """
    pred_boxes: Tensor of shape (num_queries, 4),
                each in [cx, cy, w, h] format.
    gt_box:     Tensor of shape (4,) in [cx, cy, w, h] format.

    Returns:
        iou_vals: shape (num_queries,) - IoU for each predicted box vs. the single gt_box
    """
    # pred_boxes: (N, 4) -> (N, [cx,cy,w,h])
    # Convert each [cx,cy,w,h] -> [x1, y1, x2, y2]
    cx = pred_boxes[:, 0]
    cy = pred_boxes[:, 1]
    w  = pred_boxes[:, 2]
    h  = pred_boxes[:, 3]

    x1p = cx - 0.5 * w
    y1p = cy - 0.5 * h
    x2p = cx + 0.5 * w
    y2p = cy + 0.5 * h

    # Same for the single gt_box
    cxg, cyg, wg, hg = gt_box
    x1g = cxg - 0.5 * wg
    y1g = cyg - 0.5 * hg
    x2g = cxg + 0.5 * wg
    y2g = cyg + 0.5 * hg

    # Intersection
    ix1 = torch.maximum(x1p, x1g)
    iy1 = torch.maximum(y1p, y1g)
    ix2 = torch.minimum(x2p, x2g)
    iy2 = torch.minimum(y2p, y2g)

    inter_w = torch.clamp(ix2 - ix1, min=0.0)
    inter_h = torch.clamp(iy2 - iy1, min=0.0)
    inter_area = inter_w * inter_h

    # Areas
    pred_area = (x2p - x1p) * (y2p - y1p)
    gt_area   = (x2g - x1g) * (y2g - y1g)

    union_area = pred_area + gt_area - inter_area + 1e-7  # avoid div by zero
    iou_vals = inter_area / union_area

    return iou_vals

# def match_prediction_to_gt(pred_boxes, pred_logits, gt_box, gt_class_id):
#     """
#     A naive matching function that picks the single best prediction
#     for a given ground-truth bounding box (and class) based on
#     (class_probability * IoU).
    
#     Args:
#         pred_boxes: (num_queries, 4) predicted boxes in DETR's [cx, cy, w, h] format.
#         pred_logits: (num_queries, num_classes) classification logits from DETR.
#         gt_box: (4,) ground-truth box in [cx, cy, w, h] normalized format.
#         gt_class_id: integer (the label2id for the ground-truth object).

#     Returns:
#         best_query_idx: index of the best match in [0..num_queries-1]
#         best_score: float, best (prob * IoU) found
#     """
#     # Convert logits to probabilities
#     probs = F.softmax(pred_logits, dim=-1)  # shape (num_queries, num_classes)

#     best_query_idx = None
#     best_score = -1.0

#     for q_idx in range(pred_boxes.shape[0]):
#         class_prob = probs[q_idx, gt_class_id].item()
#         iou_val = compute_iou(pred_boxes[q_idx], gt_box)
#         score = class_prob * iou_val
#         if score > best_score:
#             best_score = score
#             best_query_idx = q_idx

#     return best_query_idx, best_score

def match_prediction_to_gt(
    pred_boxes, 
    pred_logits, 
    gt_box, 
    gt_class_id, 
    prob_thresh=0.2, 
    iou_thresh=0.2
):
    """
    A vectorized matching function that picks the single best prediction
    for a given ground-truth bounding box (and class) based on:
       score = class_prob * IoU
    BUT only considers predictions where:
       class_prob >= prob_thresh AND IoU >= iou_thresh.
    
    If no predictions pass the thresholds, returns best_query_idx = -1, best_score = 0.0

    Args:
        pred_boxes:  (num_queries, 4) [cx, cy, w, h]
        pred_logits: (num_queries, num_classes)
        gt_box:      (4,) in [cx, cy, w, h]
        gt_class_id: int
        prob_thresh: float, min class probability
        iou_thresh:  float, min IoU
    
    Returns:
        best_query_idx (int): index of best match in [0..num_queries-1] or -1 if none pass
        best_score (float): best score
    """
    # 1) Convert to probabilities
    probs = F.softmax(pred_logits, dim=-1)  # (num_queries, num_classes)
    class_probs = probs[:, gt_class_id]     # (num_queries,)

    # 2) Compute IoUs for each predicted box vs. the single GT box
    iou_vals = compute_iou_batch(pred_boxes, gt_box)  # (num_queries,)

    # 3) Filter by thresholds
    valid_mask = (class_probs >= prob_thresh) & (iou_vals >= iou_thresh)

    if not valid_mask.any():
        # None pass the threshold
        return -1, 0.0

    # 4) Among the valid ones, compute score = prob * IoU
    scores = class_probs * iou_vals  # (num_queries,)

    # 5) We'll ignore the invalid ones by setting their score = -1
    masked_scores = torch.full_like(scores, -1.0)
    masked_scores[valid_mask] = scores[valid_mask]

    # 6) Argmax across all queries
    best_query_idx = torch.argmax(masked_scores)  # scalar tensor
    best_score = masked_scores[best_query_idx].item()

    return best_query_idx.item(), best_score


class ContactModel(nn.Module):
    """
    A PyTorch module that:
    - Uses a frozen DETR (from huggingface/transformers or similar) for bounding-box + class predictions.
    - Matches each ground-truth (hand, object) pair to the best predicted query.
    - Retrieves the corresponding decoder embeddings from DETR.
    - Concatenates the (hand_embedding, object_embedding) and classifies contact vs no-contact.
    """
    def __init__(self, detr, hidden_dim=256, num_contact_classes=2, max_pairs=128):
        """
        Args:
            detr: A DETR model with .forward(...) that returns classification logits,
                  predicted boxes, final decoder states, etc.
            hidden_dim: The dimension of DETR's internal transformer embeddings.
            num_contact_classes: Usually 2 for binary contact classification.
        """
        super().__init__()
        self.detr = detr
        
        # Freeze DETR parameters
        for p in self.detr.parameters():
            p.requires_grad = False

        # Create a small MLP for contact classification
        self.contact_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_contact_classes)
        )

        # We'll load the mapping from the DETR config (label2id, id2label)
        # e.g. detr.config.label2id = {"left_hand": 1, "cup": 2, ...}
        self.label2id = getattr(self.detr.config, "label2id", None)
        self.id2label = getattr(self.detr.config, "id2label", None)

        if not self.label2id or not self.id2label:
            raise ValueError(
                "DETR config must have label2id and id2label dictionaries to map class names properly."
            )
        
        self.max_pairs = max_pairs

    def forward(self, batch_data):
        pixel_values = batch_data["pixel_values"]  # (B, 3, H, W)
        pairs_batch = batch_data["pairs"]          # list of length B

        device = pixel_values.device

        # 1) Run DETR (frozen) to get predictions + final decoder embeddings
        with torch.no_grad():
            detr_outputs = self.detr(
                pixel_values,
                output_hidden_states=True,
                return_dict=True
            )

        pred_logits = detr_outputs.logits          # (B, num_queries, num_classes)
        pred_boxes  = detr_outputs.pred_boxes      # (B, num_queries, 4)
        final_decoder_hidden = detr_outputs.decoder_hidden_states[-1]  # (B, num_queries, hidden_dim)

        # Instead of storing logits/labels/weights one by one,
        # store the embeddings and other info, then do ONE pass through the MLP at the end.
        all_embeddings = []
        all_labels = []
        all_weights = []

        # 2) Loop over each sample in the batch
        for b_idx in range(pixel_values.size(0)):
            sample_boxes  = pred_boxes[b_idx]   # (num_queries, 4)
            sample_logits = pred_logits[b_idx]  # (num_queries, num_classes)
            sample_hidden = final_decoder_hidden[b_idx]  # (num_queries, hidden_dim)

            sample_pairs = pairs_batch[b_idx]  # ground-truth pairs for this image

            for pair in sample_pairs:
                label = pair["label"]  # 0 or 1
                hand_name = pair["hand_name"]
                obj_name  = pair["obj_name"]

                # Convert [cx, cy, w, h] to tensors
                hand_box = torch.tensor(pair["hand_box"], dtype=torch.float, device=device)
                obj_box  = torch.tensor(pair["obj_box"],  dtype=torch.float, device=device)

                # Map names -> class IDs
                if hand_name not in self.label2id or obj_name not in self.label2id:
                    continue  # skip unknown classes

                hand_class_id = self.label2id[hand_name]
                obj_class_id  = self.label2id[obj_name]

                # Match each GT box to the best DETR prediction
                hand_q_idx, _ = match_prediction_to_gt(sample_boxes, sample_logits, hand_box, hand_class_id)
                obj_q_idx,  _ = match_prediction_to_gt(sample_boxes, sample_logits, obj_box,  obj_class_id)

                # Retrieve final decoder embeddings for matched queries
                hand_emb = sample_hidden[hand_q_idx]  # (hidden_dim,)
                obj_emb  = sample_hidden[obj_q_idx]   # (hidden_dim,)

                # Concatenate into a single pair embedding
                pair_emb = torch.cat([hand_emb, obj_emb], dim=-1)  # (hidden_dim*2,)

                # Per-sample weight logic
                if label == 1:
                    w = 1.0
                else:
                    iou_val = compute_iou(hand_box, obj_box)
                    # w = 0.1 + 0.2 * iou_val  # in [0.1, 0.5]
                    w = 0.05 * (iou_val < 0.01) + 0.2 * (iou_val > 0.01)

                all_embeddings.append(pair_emb)
                all_labels.append(label)
                all_weights.append(w)

        # 3) Now we have *all* pair embeddings for the batch in lists:
        #    all_embeddings: list of Tensors (hidden_dim*2,)
        #    all_labels, all_weights: lists of floats/ints

        # Convert to Tensors
        if len(all_embeddings) == 0:
            # Edge case: no valid pairs in the entire batch
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "logits": None
            }

        embeddings_tensor = torch.stack(all_embeddings, dim=0)  # (N, hidden_dim*2)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)     # (N,)
        weights_tensor = torch.tensor(all_weights, dtype=torch.float, device=device)  # (N,)

        N = embeddings_tensor.size(0)  # total pairs
        max_pairs = self.max_pairs     # e.g. user-defined threshold

        # 4) Separate positives & negatives for random sampling
        pos_mask = (labels_tensor == 1)
        neg_mask = (labels_tensor == 0)

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        num_pos = pos_indices.size(0)
        num_neg = neg_indices.size(0)

        # If total > max_pairs, keep all positives and randomly sample negatives
        if N > max_pairs:
            # how many negative samples we can keep
            max_neg = max_pairs - num_pos

            if max_neg < 0:
                # Edge case: even positives alone exceed self.max_pairs
                # You might decide to keep them all or sample some positives as well.
                # For simplicity, let's keep the first `max_pairs` positives
                chosen_pos_idx = pos_indices[:max_pairs]
                chosen_neg_idx = neg_indices[:0]  # none
            else:
                # sample from the negative set
                neg_idx_list = neg_indices.tolist()
                sampled_neg_idx = random.sample(neg_idx_list, k=max_neg)

                chosen_pos_idx = pos_indices
                chosen_neg_idx = torch.tensor(sampled_neg_idx, device=device)

            final_indices = torch.cat([chosen_pos_idx, chosen_neg_idx], dim=0)
            # reorder final_indices so positives/negatives might be mixed
            # but not strictly necessary. We'll just leave it as is.
            final_count = final_indices.size(0)
        else:
            # no need to sample, keep them all
            final_indices = torch.arange(N, device=device)
            final_count = N

        # 5) If final_count < max_pairs, we will pad up to max_pairs
        #    and ignore the padded slots during loss.
        if final_count < max_pairs:
            # We'll create a pad of size (max_pairs - final_count)
            pad_size = max_pairs - final_count
            # We'll build an index array for "real" samples + "dummy" samples
            padded_indices = torch.cat([
                final_indices,
                torch.full((pad_size,), -1, dtype=torch.long, device=device)
            ])
            # The actual valid portion is first `final_count`
            # We'll create a mask for them
            valid_mask = (padded_indices >= 0)
            # Then we gather for embeddings / labels / weights
            # For 'dummy' indices of -1, we'll just gather zeros
            # Easiest is to do a custom gather approach:
            real_embeds = embeddings_tensor[final_indices]  # shape (final_count, hidden_dim*2)
            real_labels = labels_tensor[final_indices]
            real_weights = weights_tensor[final_indices]

            # Build a padded embeddings array
            padded_embeddings = torch.zeros((max_pairs, embeddings_tensor.size(1)), device=device)
            padded_labels     = torch.zeros((max_pairs,), dtype=torch.long, device=device)
            padded_weights    = torch.zeros((max_pairs,), dtype=torch.float, device=device)

            padded_embeddings[:final_count] = real_embeds
            padded_labels[:final_count]     = real_labels
            padded_weights[:final_count]    = real_weights

            # 6) Single pass through classifier
            logits = self.contact_classifier(padded_embeddings)  # (max_pairs, 2)

            # Filter out the padded region in the loss
            valid_logits = logits[valid_mask]
            valid_labels = padded_labels[valid_mask]
            valid_weights = padded_weights[valid_mask]

        else:
            # final_count == max_pairs or final_count > max_pairs
            # but we already truncated if it was > max_pairs
            # so final_count <= max_pairs here
            # Means final_count == max_pairs exactly OR final_count < max_pairs
            # but we handled the < max_pairs case above, so here final_count == max_pairs
            valid_logits = self.contact_classifier(embeddings_tensor[final_indices])  # (final_count, 2)
            valid_labels = labels_tensor[final_indices]  # (final_count,)
            valid_weights = weights_tensor[final_indices]  # (final_count,)

        # 7) Weighted Cross-Entropy
        if valid_logits.size(0) == 0:
            # Edge case: might happen if max_neg < 0 and we didn't keep anything
            # or some other logic. Return zero loss
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "logits": None
            }

        log_probs = F.log_softmax(valid_logits, dim=-1)  # (M, 2)
        picked_log_probs = log_probs[torch.arange(valid_logits.size(0), device=device),
                                    valid_labels]  # (M,)

        weighted_loss = -picked_log_probs * valid_weights
        loss_ce = weighted_loss.mean()

        # print((valid_labels == 1).sum(), (valid_labels == 0).sum())

        return {
            "loss": loss_ce,
            "logits": valid_logits,
            "labels": valid_labels,
        }
    

class AdvancedContactModel(nn.Module):
    """
    A more advanced contact estimator that:
      - Freezes DETR
      - Runs DETR up to the encoder, retrieving encoder hidden states
      - Builds a separate transformer decoder that uses bounding-box pairs as queries
      - The transformer's output embeddings are then classified into contact / no-contact
    """
    def __init__(
        self,
        detr,
        d_model=256,
        nhead=8,
        num_decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        num_contact_classes=2,
        max_pairs=128
    ):
        super().__init__()
        self.detr = detr
        self.d_model = d_model
        self.max_pairs = max_pairs
        
        # Freeze all DETR parameters
        for p in self.detr.parameters():
            p.requires_grad = False

        # ---------------------------------------------------------------------
        # 1) Build the new transformer decoder (queries for each (hand, object) pair)
        # ---------------------------------------------------------------------
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.contact_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # ---------------------------------------------------------------------
        # 2) Build an MLP to embed a pair of bboxes into a d_model-dim query
        #    If each bbox is 4D, then a pair is 8D.
        #    We'll embed 8->d_model. Feel free to get more sophisticated here.
        # ---------------------------------------------------------------------
        self.pair_embed = nn.Sequential(
            nn.Linear(8, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # ---------------------------------------------------------------------
        # 3) Classification head that maps the transformer's final output to
        #    contact vs no-contact (2 by default)
        # ---------------------------------------------------------------------
        self.contact_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_contact_classes)
        )

        # We expect detr.config.label2id and detr.config.id2label for name <-> id
        self.label2id = getattr(self.detr.config, "label2id", None)
        self.id2label = getattr(self.detr.config, "id2label", None)
        if not self.label2id or not self.id2label:
            raise ValueError("DETR config must have label2id and id2label")
        
        # print(self.id2label)
        self.id2label[36] = "background"
        self.label2id["background"] = 36
        
    def forward(self, batch_data, train=True):
        """
        Wrapper forward:
          - If train=True => forward_train()
          - Else => forward_test()

        `**kwargs` can pass additional arguments to forward_test (e.g. score_thresh=0.5).
        """
        if train:
            return self.forward_train(batch_data)
        else:
            return self.forward_test(batch_data)

    def forward_train(self, batch_data):
        """
        batch_data dict:
          - "pixel_values": (B, 3, H, W)
          - "pairs": a list of length B, each is a list of dicts:
               pair["hand_box"]  -> [cx, cy, w, h]
               pair["obj_box"]   -> [cx, cy, w, h]
               pair["hand_name"] -> string
               pair["obj_name"]  -> string
               pair["label"]     -> int (0 or 1)
        Returns a dict with keys:
          - "loss": the training loss (requires_grad=True)
          - "logits": raw logits for all used pairs
          - "labels": the ground-truth labels for those pairs
        """
        device = batch_data["pixel_values"].device
        pixel_values = batch_data["pixel_values"]
        pairs_batch  = batch_data["pairs"]

        # ---------------------------------------------------------------------
        # 1) Run DETR up to the encoder, freeze gradients
        #    The exact way you retrieve the encoder hidden states will differ
        #    depending on your DETR variant. HuggingFace's DETR returns them
        #    in model(...).encoder_last_hidden_state (if return_dict=True).
        # ---------------------------------------------------------------------
        with torch.no_grad():
            # Some DETR versions might need:
            # outputs = self.detr.model(pixel_values, output_hidden_states=True, return_dict=True)
            # memory = outputs.encoder_last_hidden_state  # (B, seq_len, d_model)
            # 
            # For the official HF DETR, the backbone + encoder can also be
            # something like:
            # backbone_outputs = self.detr.backbone(pixel_values)
            # memory = self.detr.transformer.encoder(...)
            # etc.
            # 
            # Here we pretend there's a single call:
            encoder_outputs = self.detr(
                pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
            # The key part is to get the last encoder state:
            memory = encoder_outputs.encoder_last_hidden_state  # shape (B, S, d_model)

            # We also need a corresponding spatial mask or positional encodings
            # if your DETR uses them. For simplicity, assume they are accessible or
            # can be derived. We'll call them "encoder_padding_mask" or something
            # similar. If your DETR does not return them, you need to adapt.
            # 
            # For demonstration, we'll skip mask complexities and assume we have
            # memory_mask = None, memory_key_padding_mask = None, etc.

        # We'll hold all pair embeddings, labels, weights for the entire batch:
        all_queries  = []
        all_labels   = []
        all_weights  = []
        batch_sizes  = []  # to keep track of how many pairs in each sample

        for b_idx in range(pixel_values.size(0)):
            sample_pairs = pairs_batch[b_idx]

            if len(sample_pairs) == 0:
                # Instead of continue, append empty placeholders so indexes line up
                empty_query  = torch.empty((0, self.d_model), device=device)
                empty_labels = torch.empty((0,), dtype=torch.long, device=device)
                empty_wts    = torch.empty((0,), dtype=torch.float, device=device)
                
                all_queries.append(empty_query)
                all_labels.append(empty_labels)
                all_weights.append(empty_wts)
                batch_sizes.append(0)
                continue

            # For each pair, we embed 2 bboxes (8D) into a single query (d_model)
            sample_queries = []
            sample_labels  = []
            sample_weights = []

            for pair in sample_pairs:
                label = pair["label"]  # 0 or 1
                hand_box = torch.tensor(pair["hand_box"], dtype=torch.float, device=device)
                obj_box  = torch.tensor(pair["obj_box"],  dtype=torch.float, device=device)

                # Weight logic – e.g. same as before
                if label == 1:
                    w = 1.0
                else:
                    # iou_val = compute_iou(hand_box, obj_box)
                    iou_val = pair["iou"]
                    w = 0.05 * (iou_val < 0.1) + 0.4 * (iou_val > 0.1)

                # Concatenate the two bboxes: shape (8,)
                pair_coords = torch.cat([hand_box, obj_box], dim=0)
                # Embed as query
                query_emb = self.pair_embed(pair_coords)  # (d_model,)

                sample_queries.append(query_emb)
                sample_labels.append(label)
                sample_weights.append(w)

            if len(sample_queries) > 0:
                sample_queries_tensor = torch.stack(sample_queries, dim=0)  # (num_pairs, d_model)
                sample_labels_tensor  = torch.tensor(sample_labels, dtype=torch.long, device=device)
                sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float, device=device)
            else:
                # No valid pairs
                sample_queries_tensor = torch.empty((0, self.d_model), device=device)
                sample_labels_tensor  = torch.empty((0,), dtype=torch.long, device=device)
                sample_weights_tensor = torch.empty((0,), dtype=torch.float, device=device)

            all_queries.append(sample_queries_tensor)
            all_labels.append(sample_labels_tensor)
            all_weights.append(sample_weights_tensor)
            batch_sizes.append(sample_queries_tensor.size(0))

        # Concatenate across batch dimension into a big list of queries
        # Then we’ll feed them into the decoder in a *batched* manner. We do that
        # by creating a "global" set of queries and then also a "per-sample" memory
        # from DETR. However, the standard PyTorch transformer expects you to feed
        # them all at once. Two main approaches:
        #   1) Loop over each sample's memory, decode queries -> not efficient
        #   2) Create a block-diagonal approach so we can process all in parallel.
        #
        # For simplicity, we'll do a loop-based approach here. If your batch size
        # is large, you may want something more advanced.

        # We'll store final outputs, final labels, final weights
        final_outputs = []
        final_labels  = []
        final_weights = []

        # print(len(all_queries))

        for b_idx in range(pixel_values.size(0)):
            # print('here', b_idx, len(all_queries))
            num_pairs = batch_sizes[b_idx]
            if num_pairs == 0:
                continue

            # memory for this sample
            sample_memory = memory[b_idx].unsqueeze(1)  # shape (S, 1, d_model) for PyTorch's (T, N, E)
            # or if your memory is (B, S, d_model), we might transpose to (S, B, d_model).
            # So let's do: memory is (S, d_model), we unsqueeze dim=1 for batch=1
            # Check how your DETR implements it. We'll assume we want (sequence, batch, d_model).
            sample_memory = sample_memory.transpose(0, 1)  # now shape (1, S, d_model) -> (S, 1, d_model)
            sample_memory = sample_memory.transpose(0, 1)  # final shape (S, 1, d_model)

            # queries shape: (num_pairs, d_model)
            sample_queries = all_queries[b_idx]  # (num_pairs, d_model)
            # We need to transform it to (num_queries, batch=1, d_model)
            sample_queries = sample_queries.unsqueeze(1)  # (num_pairs, 1, d_model)

            # PyTorch transformer expects (T, N, E), so do a transpose:
            sample_queries = sample_queries.transpose(0, 1)  # now (1, num_pairs, d_model)
            sample_queries = sample_queries.transpose(0, 1)  # now (num_pairs, 1, d_model)

            # We skip mask/padding for brevity, but you'd normally supply a key_padding_mask if needed

            # Pass through contact decoder
            #   shape of out: (num_pairs, 1, d_model)
            decoded = self.contact_decoder(
                tgt=sample_queries,               # (T, N, E)
                memory=sample_memory,             # (S, N, E)
                # Optional: tgt_mask=..., memory_key_padding_mask=..., etc.
            )

            # Squeeze the batch dimension => (num_pairs, d_model)
            decoded = decoded.squeeze(1)

            # Classify => (num_pairs, num_contact_classes)
            logits = self.contact_classifier(decoded)

            final_outputs.append(logits)
            final_labels.append(all_labels[b_idx])
            final_weights.append(all_weights[b_idx])

        if len(final_outputs) == 0:
            # No valid pairs in entire batch
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "logits": None
            }

        # Concatenate everything
        logits_cat = torch.cat(final_outputs, dim=0)  # shape (N, num_contact_classes)
        labels_cat = torch.cat(final_labels, dim=0)   # shape (N,)
        weights_cat = torch.cat(final_weights, dim=0) # shape (N,)

        # Now do the same logic as in your baseline: sample if more than max_pairs
        # 1) separate pos/neg
        N = logits_cat.size(0)
        pos_mask = (labels_cat == 1)
        neg_mask = (labels_cat == 0)

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        num_pos = pos_indices.size(0)
        num_neg = neg_indices.size(0)

        if N > self.max_pairs:
            max_neg = self.max_pairs - num_pos
            if max_neg < 0:
                # Even positives exceed max_pairs
                chosen_pos_idx = pos_indices[: self.max_pairs]
                chosen_neg_idx = neg_indices[:0]  # empty
            else:
                neg_idx_list = neg_indices.tolist()
                sampled_neg_idx = random.sample(neg_idx_list, k=max_neg)
                chosen_pos_idx = pos_indices
                chosen_neg_idx = torch.tensor(sampled_neg_idx, device=device)

            final_indices = torch.cat([chosen_pos_idx, chosen_neg_idx], dim=0)
        else:
            final_indices = torch.arange(N, device=device)

        final_count = final_indices.size(0)
        # If final_count < max_pairs, pad
        if final_count < self.max_pairs:
            pad_size = self.max_pairs - final_count
            padded_indices = torch.cat([
                final_indices,
                torch.full((pad_size,), -1, dtype=torch.long, device=device)
            ])
            valid_mask = (padded_indices >= 0)

            # Gather the real ones
            real_logits  = logits_cat[final_indices]
            real_labels  = labels_cat[final_indices]
            real_weights = weights_cat[final_indices]

            # Create padded placeholders
            padded_logits  = torch.zeros((self.max_pairs, logits_cat.size(1)), device=device)
            padded_labels  = torch.zeros((self.max_pairs,), dtype=torch.long, device=device)
            padded_weights = torch.zeros((self.max_pairs,), dtype=torch.float, device=device)

            padded_logits[:final_count]  = real_logits
            padded_labels[:final_count]  = real_labels
            padded_weights[:final_count] = real_weights

            valid_logits  = padded_logits[valid_mask]
            valid_labels  = padded_labels[valid_mask]
            valid_weights = padded_weights[valid_mask]

        else:
            valid_logits  = logits_cat[final_indices]
            valid_labels  = labels_cat[final_indices]
            valid_weights = weights_cat[final_indices]

        if valid_logits.size(0) == 0:
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "logits": None
            }

        # Weighted cross-entropy
        log_probs = F.log_softmax(valid_logits, dim=-1)
        picked_log_probs = log_probs[torch.arange(valid_logits.size(0), device=device),
                                     valid_labels]
        weighted_loss = -picked_log_probs * valid_weights
        loss_ce = weighted_loss.mean()

        return {
            "loss": loss_ce,
            "logits": valid_logits,
            "labels": valid_labels
        }
    
   
    

    def forward_test(self, batch_data):
        """
        Inference forward pass that returns "raw" detection & contact results.
        Steps:
        1) Run DETR => get class_logits, pred_boxes, encoder memory.
        2) For each image in the batch:
            - Convert class_logits to probabilities (softmax)
            - Find 'scores' = max prob (excluding background), 'labels' = argmax
            - Identify best left-hand, best right-hand (argmax among detections for that label)
            - If found, build (Q,8) with (best_hand_box, each_box) => pass to contact decoder => get (Q,) contact probabilities
            - Return a dict with:
                {
                "pred_boxes": (Q,4),
                "scores": (Q,),
                "labels": (Q,),
                "left_contact": (Q,),
                "right_contact": (Q,)
                }
            - left_contact or right_contact is all zeros if that hand not found
        """
        device = batch_data["pixel_values"].device
        pixel_values = batch_data["pixel_values"]  # shape (B, 3, H, W)
        B = pixel_values.size(0)

        # 1) Run DETR
        with torch.no_grad():
            detr_outputs = self.detr(
                pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
        class_logits = detr_outputs.logits         # (B, Q, num_classes)
        pred_boxes   = detr_outputs.pred_boxes     # (B, Q, 4) – if cxcywh, convert to xyxy below
        pred_boxes   = cxcywh_to_xyxy_tensor(pred_boxes)
        memory       = detr_outputs.encoder_last_hidden_state  # (B, S, d_model)

        # For convenience, let's get numeric IDs for left/right
        left_hand_label_id  = self.label2id.get("left_hand", None)
        right_hand_label_id = self.label2id.get("right_hand", None)

        batch_results = []

        # 2) Iterate over each image
        for b_idx in range(B):
            # If your pred_boxes is in cx,cy,w,h, convert to xyxy. Otherwise skip.
            # Example:
            # boxes_xyxy = cxcywh_to_xyxy(pred_boxes[b_idx])  # shape (Q,4)
            # We'll assume it's already XYXY for demonstration:
            boxes_xyxy = pred_boxes[b_idx]  # (Q,4)

            # shape (Q, num_classes)
            logits_per_image = class_logits[b_idx]

            # 2a) compute probabilities, ignoring the last class if it's "background"
            # Some DETR variants have background at index=-1 or last index => adjust as needed
            probs = F.softmax(logits_per_image, dim=-1)        # (Q, C)
            scores, label_ids = probs[..., :-1].max(dim=-1)     # (Q,) => best among non-bg classes

            # 2b) find best left-hand index
            left_mask  = (label_ids == left_hand_label_id) if (left_hand_label_id is not None) else None
            best_left_idx = None
            if left_mask is not None and left_mask.any():
                # Among all queries with label "left_hand", pick the highest score
                left_indices = left_mask.nonzero(as_tuple=True)[0]  # shape(M,)
                left_scores  = scores[left_indices]                 # shape(M,)
                best_m = torch.argmax(left_scores)
                best_left_idx = left_indices[best_m]

            # 2c) find best right-hand index
            right_mask = (label_ids == right_hand_label_id) if (right_hand_label_id is not None) else None
            best_right_idx = None
            if right_mask is not None and right_mask.any():
                right_indices = right_mask.nonzero(as_tuple=True)[0]
                right_scores  = scores[right_indices]
                best_m = torch.argmax(right_scores)
                best_right_idx = right_indices[best_m]

            # We'll gather (Q,4) for all detection boxes => call them "obj_boxes"
            obj_boxes = boxes_xyxy  # shape (Q,4)
            Q = obj_boxes.size(0)

            # We'll define two arrays for contact results => shape (Q,) each
            left_contact  = torch.zeros(Q, dtype=torch.float, device=device)
            right_contact = torch.zeros(Q, dtype=torch.float, device=device)

            # 2d) If we have a best left-hand, build (Q,8) => run contact
            if best_left_idx is not None:
                # shape(4,) for the best left hand
                best_left_box = boxes_xyxy[best_left_idx]  # (4,)
                # Expand => (Q,4)
                left_box_exp  = best_left_box.unsqueeze(0).expand(Q, 4)
                # Concat => (Q,8): [x1_h, y1_h, x2_h, y2_h, x1_o, y1_o, x2_o, y2_o]
                pair_xyxy_left = torch.cat([left_box_exp, obj_boxes], dim=1)  # (Q,8)

                # pass to pair_embed
                pair_emb_left = self.pair_embed(pair_xyxy_left)    # (Q, d_model)

                # decode => memory for b_idx => shape (S,d_model) => (S,1,d_model)
                mem_this = memory[b_idx].unsqueeze(1)              # (S,1,d_model)

                # reshape queries => (Q,1,d_model)
                pair_emb_left = pair_emb_left.unsqueeze(1)         # (Q,1,d_model)
                # PyTorch transformer expects (T,N,E). We'll do the dimension shuffle:
                pair_emb_left = pair_emb_left.transpose(0,1).transpose(0,1)  # back to (Q,1,d_model)

                decoded_left = self.contact_decoder(tgt=pair_emb_left, memory=mem_this)
                decoded_left = decoded_left.squeeze(1)             # => (Q,d_model)

                logits_left = self.contact_classifier(decoded_left) # => (Q,2)
                probs_left  = F.softmax(logits_left, dim=-1)        # => (Q,2)
                contact_left_scores = probs_left[:,1]               # => (Q,)
                left_contact = contact_left_scores  # shape (Q,)

            # 2e) If we have a best right-hand, build (Q,8) => run contact
            if best_right_idx is not None:
                best_right_box = boxes_xyxy[best_right_idx]
                right_box_exp  = best_right_box.unsqueeze(0).expand(Q, 4)
                pair_xyxy_right= torch.cat([right_box_exp, obj_boxes], dim=1)  # (Q,8)

                pair_emb_right = self.pair_embed(pair_xyxy_right)   # (Q,d_model)
                mem_this = memory[b_idx].unsqueeze(1)              # (S,1,d_model)
                pair_emb_right = pair_emb_right.unsqueeze(1)
                pair_emb_right = pair_emb_right.transpose(0,1).transpose(0,1)
                decoded_right   = self.contact_decoder(tgt=pair_emb_right, memory=mem_this)
                decoded_right   = decoded_right.squeeze(1)         # (Q,d_model)

                logits_right    = self.contact_classifier(decoded_right)
                probs_right     = F.softmax(logits_right, dim=-1)  # (Q,2)
                contact_right_scores = probs_right[:,1]            # (Q,)
                right_contact = contact_right_scores

            # 2f) build final raw result for this image
            # We'll keep everything on CPU if you prefer returning them as numpy or CPU Tensors,
            # but let's demonstrate returning them as CPU Tensors.
            # If you'd rather keep them on GPU, skip the detach().cpu() step.
            boxes_final  = boxes_xyxy.detach().cpu()
            scores_final = scores.detach().cpu()
            labels_final = label_ids.detach().cpu()
            left_contact_final  = left_contact.detach().cpu()
            right_contact_final = right_contact.detach().cpu()

            # Store
            batch_results.append({
                "pred_boxes": boxes_final,    # shape (Q,4)
                "scores": scores_final,       # shape (Q,)
                "labels": labels_final,       # shape (Q,)
                "left_contact": left_contact_final,   # shape (Q,)
                "right_contact": right_contact_final  # shape (Q,)
            })

        return batch_results

    @staticmethod
    def postprocess_inference(
        batch_raw_outputs,
        id2label,
        top_k=3,
        det_score_thresh=0.5,
        contact_score_thresh=0.0
    ):
        """
        Postprocess a list of raw inference results from `forward_test`.

        Each element of `batch_raw_outputs` should be a dict:
        {
            "pred_boxes":      (Q,4)  tensor of bounding boxes [x1,y1,x2,y2],
            "scores":          (Q,)   tensor of detection scores (max class prob),
            "labels":          (Q,)   tensor of label IDs,
            "left_contact":    (Q,)   tensor of contact probabilities for left hand -> each detection
            "right_contact":   (Q,)   same but for right hand
        }

        Steps:
        1) Apply `det_score_thresh` to filter out low-confidence detections.
        2) Build a "detections" list: [ (det_id, label_name, score, [x1,y1,x2,y2]), ... ].
        3) Identify best left-hand and best right-hand among the kept detections.
        4) For each hand, sort all detection queries by descending contact probability,
            pick top_k, and store them with label/box/score in "contact_dets".
        5) Return a list of postprocessed results, one per image.

        Args:
        batch_raw_outputs: list of length B, each is a dict with the keys above
        id2label: a dict mapping label_id -> label_name
        top_k (int): how many contact objects to keep per hand
        det_score_thresh (float): detection confidence threshold
        contact_score_thresh (float): contact probability threshold (optional)
        Returns:
        A list of dicts, each with:
            {
            "detections": [
                (det_id, label_name, det_score, [x1,y1,x2,y2]), ...
            ],
            "contact_dets": [
                {
                "hand_name": "left_hand",
                "hand_det_id": <int or None>,
                "hand_box": [x1,y1,x2,y2] or None,
                "contacts": [
                    {
                    "obj_det_id": <int>,
                    "obj_label": <str>,
                    "obj_box": [x1,y1,x2,y2],
                    "contact_score": <float>
                    },
                    ...
                ]
                },
                {
                "hand_name": "right_hand",
                "hand_det_id": ...
                "hand_box": ...
                "contacts": ...
                }
            ]
            }
        """

        results = []
        # We'll assume you might have a label2id or something, but here we just have id2label
        # Let's also figure out (optionally) the label_id for left / right if you'd like
        # but we can also identify them by label string. Up to you.

        # For each image in the batch
        for batch_idx, raw in enumerate(batch_raw_outputs):
            # 1) Extract raw data
            boxes_all   = raw["pred_boxes"]    # shape (Q,4)
            scores_all  = raw["scores"]        # shape (Q,)
            labels_all  = raw["labels"]        # shape (Q,)
            left_cont   = raw["left_contact"]  # shape (Q,)
            right_cont  = raw["right_contact"] # shape (Q,)

            Q = boxes_all.size(0)
            if Q == 0:
                # No detections at all
                results.append({
                    "detections": [],
                    "contact_dets": [
                        {
                        "hand_name": "left_hand",
                        "hand_det_id": None,
                        "hand_box": None,
                        "contacts": []
                        },
                        {
                        "hand_name": "right_hand",
                        "hand_det_id": None,
                        "hand_box": None,
                        "contacts": []
                        }
                    ]
                })
                continue

            # 2) Apply detection threshold
            keep_mask = (scores_all > det_score_thresh)  # shape(Q,)
            keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0]  # shape(K,)

            # If no kept detections, store empty
            if keep_indices.numel() == 0:
                results.append({
                    "detections": [],
                    "contact_dets": [
                        {
                        "hand_name": "left_hand",
                        "hand_det_id": None,
                        "hand_box": None,
                        "contacts": []
                        },
                        {
                        "hand_name": "right_hand",
                        "hand_det_id": None,
                        "hand_box": None,
                        "contacts": []
                        }
                    ]
                })
                continue

            # Slice out the kept detection boxes, labels, scores
            det_ids  = keep_indices
            det_boxes = boxes_all[det_ids]    # (K,4)
            det_lbls  = labels_all[det_ids]   # (K,)
            det_scrs  = scores_all[det_ids]   # (K,)
            # We'll keep the corresponding contact arrays
            det_left_cont  = left_cont[det_ids]   # (K,)
            det_right_cont = right_cont[det_ids]  # (K,)

            # Build a list for final detections
            # [ (det_id, label_name, score, [x1,y1,x2,y2]), ... ]
            det_list = []
            for i in range(det_ids.size(0)):
                q_id   = det_ids[i].item()
                score  = det_scrs[i].item()
                lbl_id = det_lbls[i].item()
                label_str = id2label.get(lbl_id, f"id_{lbl_id}")
                box_xyxy  = det_boxes[i].tolist()  # [x1,y1,x2,y2]

                det_list.append((q_id, label_str, score, box_xyxy))

            # 3) Identify best left-hand among these kept detections
            #    We'll just pick the detection with label == "left_hand" with highest score
            #    or you can rely on your known label_id for "left_hand"
            best_left_idx = None
            best_left_scr = -1.0
            best_left_box = None
            best_left_id  = None

            best_right_idx = None
            best_right_scr = -1.0
            best_right_box = None
            best_right_id  = None

            # For each detection, if it's "left_hand", see if we have a better score, etc.
            for i in range(det_ids.size(0)):
                q_id   = det_ids[i].item()
                lbl_id = det_lbls[i].item()
                lbl_str= id2label.get(lbl_id, f"id_{lbl_id}")
                scr    = det_scrs[i].item()
                box4   = det_boxes[i]

                if lbl_str == "left_hand" and scr > best_left_scr:
                    best_left_scr = scr
                    best_left_idx = i
                    best_left_box = box4
                    best_left_id  = q_id

                if lbl_str == "right_hand" and scr > best_right_scr:
                    best_right_scr = scr
                    best_right_idx = i
                    best_right_box = box4
                    best_right_id  = q_id

            # 4) For each hand, we want to sort the contact array descending
            #    and pick top_k. Then gather box/label for each contact object.
            
            def get_topk_contacts(hand_idx, contact_arr):
                """
                Sort 'contact_arr' in descending order. Return top_k items.
                Each item is dict with obj_det_id, obj_label, obj_box, contact_score.
                We also can optionally apply 'contact_score_thresh' if desired.
                """
                if hand_idx is None:
                    # No such hand => return empty
                    return []

                # contact_arr shape (K,)
                sort_idx = torch.argsort(contact_arr, descending=True)
                # Keep only top_k
                sort_idx = sort_idx[:top_k]
                
                contacts_list = []
                for s_i in sort_idx:
                    cscore = contact_arr[s_i].item()
                    if cscore < contact_score_thresh:
                        # if you want to filter out low contact prob, break or continue
                        break
                    obj_qid   = det_ids[s_i].item()
                    obj_lblid = det_lbls[s_i].item()
                    obj_label = id2label.get(obj_lblid, f"id_{obj_lblid}")
                    obj_box   = det_boxes[s_i].tolist()

                    contacts_list.append({
                        "obj_det_id":    obj_qid,
                        "obj_label":     obj_label,
                        "obj_box":       obj_box,
                        "contact_score": cscore
                    })
                return contacts_list

            left_contacts_info = get_topk_contacts(best_left_idx, det_left_cont)
            right_contacts_info= get_topk_contacts(best_right_idx, det_right_cont)

            # 5) Build the final "contact_dets" structure
            def to_list(box_tensor):
                return box_tensor.tolist() if (box_tensor is not None) else None

            contact_dets = [
                {
                "hand_name": "left_hand",
                "hand_det_id": best_left_id,
                "hand_box": to_list(best_left_box),
                "contacts": left_contacts_info
                },
                {
                "hand_name": "right_hand",
                "hand_det_id": best_right_id,
                "hand_box": to_list(best_right_box),
                "contacts": right_contacts_info
                }
            ]

            # 6) Append final result
            results.append({
                "detections": det_list,
                "contact_dets": contact_dets
            })

        return results


    def save(self, save_path):
        full_state_dict = self.state_dict()
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items()
            if not "detr" in k
        }
        torch.save(filtered_state_dict, save_path)
        print(f"Contact model weights (excluding DETR) saved to {save_path}")
# models/detector.py
"""
Zombie detector: Faster R-CNN with a frozen ResNet-50 FPN backbone.

Theory:
  FPN (Feature Pyramid Network, Lin et al. 2017) attaches lateral connections
  at five backbone stages and upsamples them into a pyramid of feature maps.
  The smallest level (P2, stride=4) produces anchors at 32 px — well-matched
  to our zombies (~29×31 px on a 720×1280 canvas).  Without FPN a standard
  CNN would miss objects this small after multiple pooling layers.

  We freeze the ResNet-50 backbone (pre-trained on ImageNet) and only train
  the Region Proposal Network and the RoI head.  This follows the privileged-
  learning paradigm of Chen et al. (Learning by Cheating, 2020): learn cheap
  features fast, then bridge to the target observation space.

Public API:
  build_detector(frozen_backbone=True) -> nn.Module
  predict(model, image_np, threshold, device) -> (N,4) float32 [l,t,w,h]
"""

import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 0 = background (reserved by Faster R-CNN), 1 = zombie
NUM_CLASSES = 2


def build_detector(
    frozen_backbone: bool = True,
    min_size: int = 480,
    max_size: int = 640,
) -> torch.nn.Module:
    """
    Return a Faster R-CNN model ready for fine-tuning on KAZ zombie frames.

    Args:
        frozen_backbone: Freeze ResNet-50 FPN weights; only RPN + RoI head
                         are trained.  Recommended: faster convergence, lower
                         GPU memory, fits the 30 h/week Kaggle budget.
        min_size / max_size: Internal resize limits for the FPN transform.
                         Default 480/640 caps the 720×1280 render at
                         480×640, keeping zombies (~14×15 px at this scale)
                         well above the P2 anchor size (32 px stride-4) while
                         cutting per-image forward time by ~3× vs the torchvision
                         default (min 800, max 1333).
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=min_size, max_size=max_size)

    if frozen_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Swap the 91-class COCO head for a 2-class (background + zombie) head.
    # Only the new head parameters are initialised randomly and will be trained.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model


def save_heads_fp16(model: torch.nn.Module, path: str) -> None:
    """Save only the fine-tuned heads (RPN + roi_heads) in FP16.

    The frozen backbone is omitted — it is identical to torchvision's
    pretrained COCO weights and will be re-downloaded at load time.
    Saves ~32 MB vs ~165 MB for the full model.
    """
    state = {
        "rpn":       {k: v.half() for k, v in model.rpn.state_dict().items()},
        "roi_heads": {k: v.half() for k, v in model.roi_heads.state_dict().items()},
    }
    torch.save(state, path)


def load_detector_heads(
    path: str,
    device: str = "cpu",
    min_size: int = 480,
    max_size: int = 640,
) -> torch.nn.Module:
    """Load a heads-only FP16 checkpoint produced by save_heads_fp16().

    Builds a fresh model (downloads backbone from torchvision cache if needed),
    then overwrites RPN and roi_heads with the fine-tuned weights.
    """
    model = build_detector(frozen_backbone=True, min_size=min_size, max_size=max_size)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.rpn.load_state_dict(
        {k: v.float() for k, v in ckpt["rpn"].items()}
    )
    model.roi_heads.load_state_dict(
        {k: v.float() for k, v in ckpt["roi_heads"].items()}
    )
    model.eval().to(device)
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_np: np.ndarray,
    threshold: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run inference on a single (H, W, 3) uint8 numpy image.

    Returns:
        (N, 4) float32 array in [left, top, width, height] pixel coordinates,
        matching the format expected by submission.CustomZombieDetectorFunction
        and the iou() function in utils.py.  Returns shape (0, 4) if no zombie
        exceeds the confidence threshold.
    """
    model.eval()
    model.to(device)

    # HWC uint8 → CHW float32 in [0, 1]
    img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    output = model([img_t.to(device)])[0]

    keep = output["scores"] >= threshold
    boxes_xyxy = output["boxes"][keep].cpu().numpy()  # (N, 4) x1y1x2y2

    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Convert x1y1x2y2 → left top width height
    ltwh = boxes_xyxy.copy()
    ltwh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # width
    ltwh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # height
    return ltwh.astype(np.float32)

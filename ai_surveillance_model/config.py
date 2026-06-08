from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_path: str = "models/yolo11n.pt"
    device: Optional[str] = None
    conf_threshold: float = 0.5
    iou_threshold: float = 0.3
    max_lost_frames: int = 5
    loitering_seconds: float = 30.0
    crowd_threshold: int = 2
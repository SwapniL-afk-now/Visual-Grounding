import torch
from PIL import Image
from ultralytics import SAM
import numpy as np

class SAM3Teacher:
    """
    SAM3 (Segment Anything Model 3) Wrapper to serve as a 'Teacher' for grounding.
    Provides precise object masks and bounding boxes.
    """
    def __init__(self, model_id="sam2_b.pt"): # Using SAM2/3 weights via Ultralytics
        print(f"Initializing SAM3 Teacher ({model_id})...")
        self.model = SAM(model_id)
        
    def get_ground_truth(self, image_path, query_box=None, point_coords=None):
        """
        Produce a high-precision mask and box for the target.
        """
        results = self.model.predict(
            source=image_path,
            bboxes=query_box,
            points=point_coords,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if not results or len(results[0].masks) == 0:
            return None, None
            
        # Get the highest confidence mask
        mask = results[0].masks.data[0].cpu().numpy()
        box = results[0].boxes.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
        
        return mask, box

# Example integration for Reward Function
def calculate_teacher_reward(vlm_box, vlm_attention, teacher_mask, teacher_box):
    # 1. IoU Component
    iou = calculate_giou(vlm_box, teacher_box)
    
    # 2. Focus Component (Attention inside SAM3 mask)
    # Resize mask to attention grid size
    focus = (vlm_attention * teacher_mask_resized).sum() / vlm_attention.sum()
    
    return iou * focus

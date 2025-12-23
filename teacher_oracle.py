import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM

class TeacherOracle:
    """
    Teacher Oracle using Grounding DINO (Text-to-Box) for high-precision grounding truth.
    """
    def __init__(self, dino_id="IDEA-Research/grounding-dino-tiny"):
        print(f"Initializing Teacher Oracle (DINO: {dino_id})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Grounding DINO for text-to-box
        self.dino_processor = AutoProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device).to(torch.float16)

    def get_ground_truth(self, image, text_query, box_threshold=0.3, text_threshold=0.25):
        """
        Takes an image and a text query, returns Grounding DINO's predicted BBox.
        """
        inputs = self.dino_processor(images=image, text=text_query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        if len(results["boxes"]) == 0:
            return None

        # Take the most confident box
        best_idx = results["scores"].argmax()
        dino_box = results["boxes"][best_idx].cpu().numpy() # [x1, y1, x2, y2]
        
        return dino_box

def compare_grounding(vlm_box, teacher_box):
    """Utility to calculate GIoU between VLM and Teacher."""
    # Logic for IoU calculation goes here
    pass

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from scipy.ndimage import zoom

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "animals.jpg"  # Default fallback

SYSTEM_PROMPT = (
    "You are a high-precision visual grounding expert. "
    "Follow this exact sequence for every response:\n"
    "1. <think>: Reasoning about object locations and spatial relationships.\n"
    "2. <query>: A simplified, concise name of the object(s) found (for external grounding tools).\n"
    "3. <answer>: Coordinates in [xmin, ymin, xmax, ymax] format (0-1000 scale).\n\n"
    "Example:\n"
    "<think>The red cup is on the wooden table next to the laptop.</think>\n"
    "<query>red cup</query>\n"
    "<answer>[120, 450, 300, 680]</answer>"
)

def load_model():
    print(f"Loading {MODEL_ID} (Dual GPU, Eager Attention)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor

def run_inference(model, processor, image_path, query):
    print(f"Processing: {image_path} with query: '{query}'")
    image = Image.open(image_path)
    width, height = image.size
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"The image has a resolution of {width}x{height}. {query} Provide coordinates in [xmin, ymin, xmax, ymax] format inside <answer> tags."}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False 
        )
        generated_ids = outputs.sequences[:, inputs.input_ids.size(1):]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_tokens = [processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids[0]]
    
    # ... (indexing logic remains the same)
    input_ids = inputs.input_ids[0].cpu().tolist()
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    vis_start, vis_end = None, None
    for i, tid in enumerate(input_ids):
        if tid == vision_start_id:
            vis_start = i + 1
        elif tid == vision_end_id and vis_start is not None:
            vis_end = i
            break
    
    grid_t, grid_h, grid_w = inputs['image_grid_thw'][0].cpu().numpy()
    t_h, t_w = int(grid_h // 2), int(grid_w // 2)
    
    # Robust parsing
    think_indices = []
    answer_indices = []
    full_str = ""
    for i, tok in enumerate(generated_tokens):
        full_str += tok
        last_think_open = full_str.rfind("<think>")
        last_think_close = full_str.rfind("</think>")
        last_answer_open = full_str.rfind("<answer>")
        last_answer_close = full_str.rfind("</answer>")
        
        in_think = (last_think_open > last_think_close)
        in_answer = (last_answer_open > last_answer_close)
        is_marker = any(m in tok for m in ["<", ">", "think", "answer", "/"])
        
        if in_think and not is_marker:
            think_indices.append(i)
        elif in_answer and not is_marker:
            answer_indices.append(i)
    
    def get_hm(indices, attns, v_start, v_end, h, w):
        if not indices or v_start is None: return None
        total_attn = None
        count = 0
        for idx in indices:
            if idx >= len(attns): continue
            step_layers = [l[0].mean(dim=0).cpu() for l in attns[idx]]
            step_avg = torch.stack(step_layers).mean(dim=0)
            row = step_avg[-1, :]
            if total_attn is None:
                total_attn = row
            else:
                L = min(len(total_attn), len(row))
                total_attn = total_attn[:L] + row[:L]
            count += 1
        if total_attn is None or count == 0: return None
        img_part = total_attn[v_start:v_end]
        if len(img_part) != h*w:
            side = int(np.sqrt(len(img_part)))
            if side*side == len(img_part): h, w = side, side
            else: return None
        hm = img_part.view(h, w).float().numpy()
        return (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    think_hm = get_hm(think_indices, outputs.attentions, vis_start, vis_end, t_h, t_w)
    answer_hm = get_hm(answer_indices, outputs.attentions, vis_start, vis_end, t_h, t_w)
    
    return output_text, image, think_hm, answer_hm

import supervision as sv

def visualize_result(output_text, image, think_heatmap, answer_heatmap, query, teacher_box=None):
    print(f"\nModel Output:\n{output_text}")
    
    # 1. Extraction (Coordinates and Query)
    answer_section = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    search_text = answer_section.group(1) if answer_section else output_text
    nums = re.findall(r"(\d+)", search_text)
    
    query_tag_match = re.search(r"<query>(.*?)</query>", output_text, re.DOTALL)
    extracted_query = query_tag_match.group(1).strip() if query_tag_match else query
    
    # Group into [xmin, ymin, xmax, ymax] boxes
    raw_boxes = [[float(n) for n in nums[i:i+4]] for i in range(0, len(nums) - len(nums) % 4, 4)]
    
    # Scale to Pixels
    w, h = image.size
    pixel_boxes = []
    labels = []
    for i, box in enumerate(raw_boxes):
        xmin, ymin, xmax, ymax = box
        px_box = [
            xmin * w / 1000, 
            ymin * h / 1000, 
            xmax * w / 1000, 
            ymax * h / 1000
        ]
        pixel_boxes.append(px_box)
        labels.append(f"OBJ {i+1}")

    # 2. Professional Annotation using 'supervision'
    image_np = np.array(image)
    if pixel_boxes:
        detections = sv.Detections(xyxy=np.array(pixel_boxes))
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_padding=6,
            text_scale=0.8
        )
        annotated_frame = box_annotator.annotate(scene=image_np.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    else:
        annotated_frame = image_np

    # 3. Visualization Layout
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    
    # Panel 1: VLM Precision Boxes
    axes[0].imshow(annotated_frame)
    axes[0].set_title(f"VLM Prediction: '{extracted_query}'", fontsize=14)
    axes[0].axis("off")

    # Panel 2 & 3: Attention Maps
    for idx, (hm, title, cmap) in enumerate([(think_heatmap, "Panel 2: <think> Attention", "jet"), 
                                            (answer_heatmap, "Panel 3: <answer> Attention", "hot")]):
        ax = axes[idx+1]
        if hm is not None:
            zh, zw = h / hm.shape[0], w / hm.shape[1]
            h_res = zoom(hm, (zh, zw), order=1)
            ax.imshow(image)
            ax.imshow(h_res, cmap=cmap, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "Heatmap Unavailable", ha='center', va='center')
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    # Panel 4: Teacher Oracle (Grounding DINO)
    ax_teacher = axes[3]
    if teacher_box is not None:
        import supervision as sv
        t_detections = sv.Detections(xyxy=np.array([teacher_box]))
        t_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        t_label_annotator = sv.LabelAnnotator(color=sv.Color.RED)
        
        t_frame = np.array(image)
        t_frame = t_annotator.annotate(scene=t_frame, detections=t_detections)
        t_frame = t_label_annotator.annotate(scene=t_frame, detections=t_detections, labels=["DINO TRUTH"])
        
        ax_teacher.imshow(t_frame)
    else:
        ax_teacher.imshow(image)
        ax_teacher.text(0.5, 0.5, "DINO Failed", color='red', ha='center', va='center', fontsize=20)
    ax_teacher.set_title(f"Teacher Oracle (DINO)", fontsize=14)
    ax_teacher.axis("off")

    plt.tight_layout()
    save_path = "grounding_diagnostic.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Result saved to {save_path}")
    
    # Force display in interactive environments
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display, Image as IPImage
            display(IPImage(filename=save_path))
    except ImportError:
        pass

if __name__ == "__main__":
    import argparse
    from teacher_oracle import TeacherOracle
    
    parser = argparse.ArgumentParser(description="Multi-purpose Visual Grounding Diagnostic with DINO Teacher")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("query", help="Text query (e.g., 'Find the box on the table')")
    parser.add_argument("--teacher", action="store_true", help="Enable Grounding DINO Teacher Oracle")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        sys.exit(1)
    
    model, processor = load_model()
    out_text, img, t_hm, a_hm = run_inference(model, processor, args.image, args.query)
    
    # Extract Teacher Ground Truth if enabled
    t_box = None
    if args.teacher:
        # Extract <query> tag from VLM response
        query_tag_match = re.search(r"<query>(.*?)</query>", out_text, re.DOTALL)
        oracle_query = query_tag_match.group(1).strip() if query_tag_match else args.query
        
        oracle = TeacherOracle()
        t_box = oracle.get_ground_truth(img, oracle_query)
        
    visualize_result(out_text, img, t_hm, a_hm, args.query, teacher_box=t_box)

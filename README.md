# Visual Grounding & Attention Diagnostic for Qwen2.5-VL

A high-precision visual grounding and attention visualization tool for Qwen2.5-VL models. This tool extracts internal cross-attention heatmaps during both the **Reasoning (Think)** and **Decision (Answer)** phases to diagnose the "Grounding Gap" in Vision-Language Models.

![Visualization Sample](think_answer_diagnostic.png)

## Core Features
- **Generic Grounding**: Locate any object in any image using natural language.
- **Attention Diagnostics**: Extract and visualize separate heatmaps for `<think>` and `<answer>` phases.
- **SAM3 Teacher Oracle**: Ready-to-use wrapper for high-precision segmentation masks and BBox refinement.
- **High-Precision Visualization**: Powered by the `supervision` library with automatic color-cycling and labels.
- **Interactive Display**: Automatic `%matplotlib` aware image display for Jupyter/Kaggle environments.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SwapniL-afk-now/Visual-Grounding.git
   cd Visual-Grounding
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment or `uv` for faster installation:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Grounding Diagnostic
Run the diagnostic script by providing an image path and a text query:

```bash
python final_grounding_diagnostic.py "path/to/image.jpg" "Find the white bear on the left"
```

### 2. SAM3 Teacher Oracle
To use the SAM3 teacher for mask generation or reward calculation:

```python
from sam3_teacher import SAM3Teacher

teacher = SAM3Teacher()
mask, precise_box = teacher.get_ground_truth("image.jpg", query_box=[xmin, ymin, xmax, ymax])
```

## How it Works
The script leverages **Grad-CAM attention aggregation** across all layers of the Qwen2.5-VL model. It maps the visual attention tokens specifically associated with the reasoning process (`<think>`) and the final numeric output (`<answer>`).

## Requirements
- Python 3.10+
- PyTorch 2.5+
- Transformers (from source suggested for latest Qwen2.5-VL support)
- 16GB+ VRAM suggested (Dual T4 or Single A100/L40)

## License
MIT

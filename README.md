# Visual Grounding & Attention Diagnostic for Qwen2.5-VL

A high-precision visual grounding and attention visualization tool for Qwen2.5-VL models. This tool extracts internal cross-attention heatmaps during both the **Reasoning (Think)** and **Decision (Answer)** phases to diagnose the "Grounding Gap" in Vision-Language Models.

![Visualization Sample](think_answer_diagnostic.png)

## Core Features
- **Generic Grounding**: Locate any object in any image using natural language.
- **Attention Diagnostics**: Extract and visualize separate heatmaps for `<think>` and `<answer>` phases.
- **Multi-Object Parsing**: Robustly handles multiple bounding boxes and dense coordinate outputs.
- **Clean XML Output**: Enforces structured reasoning via `<think>...</think><answer>...</answer>` tags.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/visual-grounding-diagnostic.git
   cd visual-grounding-diagnostic
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment or `uv` for faster installation:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the diagnostic script by providing an image path and a text query:

```bash
python final_grounding_diagnostic.py "path/to/image.jpg" "Find the white bear on the left"
```

### Script Arguments:
- `image`: Path to the input image.
- `query`: The text query describing what to locate.

## How it Works

The script leverages **Grad-CAM attention aggregation** across all layers of the Qwen2.5-VL model. It maps the visual attention tokens specifically associated with the reasoning process (`<think>`) and the final numeric output (`<answer>`), allowing researchers to see if the model is "looking" at the right spot even if its numeric coordinates are slightly off.

## Requirements
- Python 3.10+
- PyTorch 2.5+
- Transformers (from source suggested for latest Qwen2.5-VL support)
- 16GB+ VRAM suggested (Dual T4 or Single A100/L40)

## License
MIT

# Visual Grounding & Attention Diagnostic for Qwen2.5-VL

A high-precision visual grounding and attention visualization tool for Qwen2.5-VL models. This tool extracts internal cross-attention heatmaps during both the **Reasoning (Think)** and **Decision (Answer)** phases to diagnose the "Grounding Gap" in Vision-Language Models.

![Visualization Sample](think_answer_diagnostic.png)

## Core Features
- **Generic Grounding**: Locate any object in any image using natural language.
- **Attention Diagnostics**: Extract and visualize separate heatmaps for `<think>` and `<answer>` phases.
- **Teacher Oracle Loop**: Uses Grounding DINO and SAM to provide objective BBox and Mask "truth".
- **4-Panel Visualization**: Compare VLM Prediction, reasoning attention, and Teacher Oracle verdict side-by-side.
- **Teacher Oracle Loop**: Uses Grounding DINO to provide objective BBox "truth".
- **4-Panel Visualization**: Compare VLM Prediction, reasoning attention, and Grounding DINO verdict side-by-side.
- **High-Precision Visualization**: Powered by the `supervision` library with automatic color-cycling.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SwapniL-afk-now/Visual-Grounding.git
   cd Visual-Grounding
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Grounding Diagnostic (3 Panels)
Run the standard diagnostic to see VLM internal attention:
```bash
python final_grounding_diagnostic.py "image.jpg" "Find the dog"
```

### 2. Teacher Oracle Mode (4 Panels)
Run with the `--teacher` flag to activate Grounding DINO for objective verification:
```bash
python final_grounding_diagnostic.py "image.jpg" "Find the dog" --teacher
```

## How it Works
1. **VLM Generation**: Qwen2.5-VL generates `<think>` (reasoning), `<query>` (target name), and `<answer>` (grounding).
2. **Teacher Verification**: The `<query>` tag is passed to **Grounding DINO** to find the objective bounding box.
3. **Comparison**: All signals are visualized in a 4-panel plot to diagnose the "Grounding Gap."

## Requirements
- Python 3.10+
- PyTorch 2.5+
- Transformers (from source suggested for latest Qwen2.5-VL support)
- 16GB+ VRAM suggested (Dual T4 or Single A100/L40)

## License
MIT

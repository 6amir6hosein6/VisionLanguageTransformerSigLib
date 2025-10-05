# VisionLanguageTransformerPaliGemma

This project implements a **Vision-Language Model (VLM)** for generating text from images. It uses:

- **SigLib Encoder** – for contrastive image encoding.
- **Paligemma-3B-PT-224 Decoder** – for text generation.

---

## Description

This model takes an input image and generates descriptive text. It combines a contrastive encoder with a large language model decoder.  

---

## Installation

1. Clone the repository:

```bash
git clone git@github.com:6amir6hosein6/VisionLanguageTransformerPaligemma.git
cd vlm-image2text
```
2. Install dependencies:

```bash
git pip install -r requirements.txt
```

3. Download the Paligemma-3B-PT-224 weights from Hugging Face 

```bash
hf download google/paligemma-3b-pt-224
```

Place them in:

```bash
MODEL_PATH="$HOME/model/paligemma-weights/paligemma-3b-pt-224"
```

## Usage

Option 1: Set the parameters in launch_inference.sh and Run the launch script

```bash 
#!/bin/bash
MODEL_PATH="$HOME/model/paligemma-weights/paligemma-3b-pt-224"
PROMPT="Where is this"
IMAGE_FILE_PATH="test_images/tet_image.png"
MAX_TOKENS_TO_GENERATE=50
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"
```
Option 2: Run the Python file directly with parameters
```bash
python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU
```

## Example Table for Results

| Image | Question (Prefix) | Answer (Suffix) |
|-------|-----------------|----------------|
| test_images/pic1.jpeg | this building is | [Generated text] |
| test_images/pic2.jpeg | a cat sitting on | [Generated text] |
| test_images/pic3.jpeg | the vehicle is | [Generated text] |


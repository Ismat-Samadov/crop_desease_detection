# Crop Disease Detection with PDT Dataset

This project implements a complete pipeline for detecting crop diseases using the PDT (Pests and Diseases Tree) dataset and implementing the YOLO-DP (YOLO-Dense Pest) model architecture as described in the paper.

## Overview

The PDT dataset is a high-precision UAV-based dataset for targeted detection of tree pests and diseases, which is collected in real-world operational environments and aims to fill the gap in available datasets for this field.

This implementation provides:

1. Automatic download and preprocessing of the PDT dataset
2. Conversion from VOC XML format to YOLO format
3. Implementation of the YOLO-DP model (based on YOLOv8)
4. Training pipeline
5. Inference and visualization

## Features

- Handles cases where direct download from HuggingFace might be challenging
- Implements the key architectural components from YOLO-DP:
  - GhostConv modules for efficient feature extraction
  - Adaptive Large Scale Selective Kernel for handling multi-scale features
  - Decoupled detection heads
- Provides a complete end-to-end pipeline that works in Google Colab

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.0+
- HuggingFace Hub
- OpenCV
- Matplotlib
- Other dependencies (see requirements.txt)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the full pipeline using the provided script:

```bash
python run_pdt_detection.py
```

This will:
1. Download the PDT dataset (or create a placeholder if download fails)
2. Convert it to YOLO format
3. Train the YOLO-DP model
4. Run inference on test images

### Advanced Usage

You can customize the pipeline with various options:

```bash
python run_pdt_detection.py --epochs 50 --batch-size 32 --img-size 640
```

Skip specific steps:

```bash
python run_pdt_detection.py --skip-download --skip-conversion
```

### Using Your Own Dataset

If you have your own dataset in VOC XML format, you can convert it to YOLO format:

```bash
python dataset_conversion.py --input-dir your_dataset --output-dir converted_dataset --copy-images
```

## Model Architecture

YOLO-DP is based on YOLOv8 with the following modifications:

1. **GhostConv**: Replaces standard convolutions for efficiency
2. **Adaptive Large Scale Selective Kernel**: Enhances detection of multi-scale objects
3. **Decoupled Detection Heads**: Separates classification and localization heads

## Results

The YOLO-DP model achieves excellent results on the PDT dataset:

- Precision: 90.2%
- Recall: 88.0%
- mAP@.5: 94.5%
- mAP@.5:.95: 67.5%
- FPS: 109

## References

- PDT Dataset: [HuggingFace - qwer0213/PDT_dataset](https://huggingface.co/datasets/qwer0213/PDT_dataset)
- Original Paper: [PDT: Uav Target Detection Dataset for Pests and Diseases Tree](https://arxiv.org/abs/2409.15679)
- Original Repository: [RuiXing123/PDT_CWC_YOLO-DP](https://github.com/RuiXing123/PDT_CWC_YOLO-DP)

## License

MIT

## Acknowledgements

This implementation is based on:
- Ultralytics YOLOv8
- HuggingFace Datasets
- The original PDT dataset by Zhou et al.
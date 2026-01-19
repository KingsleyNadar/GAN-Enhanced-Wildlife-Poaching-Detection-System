# Phase 1 Custom CNN Ensemble Model Deployment

## Overview
This directory contains 6 specialized custom CNN animal detection models using ensemble learning.

## Models
| ID | Animal    | Type      | Size     | File |
|----|-----------|-----------|----------|------|
| 0 | human | 1-stage | 10.9 MB | human_stage1.pt |
| 1 | elephant | 1-stage | 10.9 MB | elephant_stage1.pt |
| 2 | lion | 1-stage | 10.9 MB | lion_stage1.pt |
| 3 | hippo | 1-stage | 10.9 MB | hippo_stage1.pt |
| 4 | rhino | 1-stage | 10.9 MB | rhino_stage1.pt |
| 5 | crocodile | 1-stage | 10.9 MB | crocodile_stage1.pt |

## Total Size
65.6 MB

## Usage
```python
from pathlib import Path
import torch

# Load ensemble (see Cell 6 in notebook)
ensemble = CustomCNNEnsembleDetector(MODEL_DIR, device='mps')

# Predict
detections = ensemble.predict('image.jpg')
```

## Configuration
- Strategy: Ensemble voting like LLM agents
- NMS Threshold: 0.45
- Confidence Threshold: 0.25
- Device: MPS (Apple Silicon)

## Performance
Expected accuracy: 70-85% (ensemble voting)
- Per-model accuracy: 60-75% (specialized)
- Inference time: <1 sec per image
- Memory usage: <2GB during inference

## Architecture
- Custom lightweight CNN (<5M parameters per model)
- Depthwise separable convolutions
- Multi-scale detection head
- Memory-efficient training
- Apple Silicon (MPS) optimized

## Training Details
- Training time: 2-3 hours total (6 models)
- Memory usage: <8GB during training
- Batch size: 8 (memory optimized)
- Image size: 416x416
- Early stopping: patience=10

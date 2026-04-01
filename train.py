"""
PSD-RTDETR training example.

Before running, ensure you have:
1. Installed dependencies: pip install -r requirements.txt
2. Integrated PSD-RTDETR modules into ultralytics (see docs/integration.md)
3. Prepared your dataset in YOLO format (see datasets/example.yaml)
"""

import warnings
import os

warnings.filterwarnings('ignore')

# Set GPU device (change to "0", "1", etc. for GPU training)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/PSD-RTDETR.yaml')
    model.train(
        data='datasets/example.yaml',  # your dataset config
        imgsz=640,
        epochs=200,
        batch=8,
        device='0',          # '0' for first GPU, 'cpu' for CPU
        workers=4,           # set to 0 on Windows if data loading hangs
        project='runs/train',
        name='exp',
    )

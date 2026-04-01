# Integrating PSD-RTDETR Modules into Ultralytics

This guide explains how to integrate the three proposed modules (PCGLU, AIFI_SEFFN, DSFD) into the [Ultralytics](https://github.com/ultralytics/ultralytics) framework (tested with v8.0.201).

## Prerequisites

```bash
pip install ultralytics==8.0.201 timm
```

## Step 1: Copy Module Files

Copy the module files from `models/` into the ultralytics package:

```bash
# Find your ultralytics installation path
python -c "import ultralytics; print(ultralytics.__file__)"
# e.g., /path/to/site-packages/ultralytics/__init__.py

# Create extra_modules directory if it doesn't exist
mkdir -p /path/to/site-packages/ultralytics/nn/extra_modules/

# Copy module files
cp models/pcglu.py   /path/to/site-packages/ultralytics/nn/extra_modules/block.py
cp models/seffn.py   /path/to/site-packages/ultralytics/nn/extra_modules/transMamba.py
cp models/aifi_seffn.py /path/to/site-packages/ultralytics/nn/extra_modules/transformer.py
cp models/dsfd.py    /path/to/site-packages/ultralytics/nn/extra_modules/hcfnet.py
```

## Step 2: Update Imports in Module Files

After copying, update the internal imports to match the ultralytics package structure:

**In `block.py` (PCGLU):**

The standalone version uses a self-contained `Faster_Block_CGLU`. For full integration, make `PCGLU` inherit from the ultralytics `BasicBlock`:

```python
# Change the import at the top
from ..modules.conv import Conv
from ..modules.block import BasicBlock

# Change the PCGLU class to inherit from BasicBlock
class PCGLU(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.branch2b = Faster_Block_CGLU(ch_out, ch_out)
```

This replaces the second 3x3 conv in BasicBlock with `Faster_Block_CGLU` while keeping the residual shortcut and first conv from the parent class.

**In `transformer.py` (AIFI_SEFFN):**

```python
# Change the import
from .transMamba import SpectralEnhancedFFN
```

And add the torch version check:

```python
from ...utils.torch_utils import TORCH_1_9
if not TORCH_1_9:
    raise ModuleNotFoundError(
        'TransformerEncoderLayer() requires torch>=1.9 to use '
        'nn.MultiheadAttention(batch_first=True).')
```

## Step 3: Create `__init__.py`

Create `ultralytics/nn/extra_modules/__init__.py`:

```python
from .transformer import *
from .block import *
from .hcfnet import *
```

## Step 4: Import in `tasks.py`

Add this import to `ultralytics/nn/tasks.py` (near the top, with other imports):

```python
from ultralytics.nn.extra_modules import *
```

## Step 5: Register Modules in `parse_model()`

In the `parse_model()` function in `ultralytics/nn/tasks.py`, add handling for the new modules inside the module-type conditional blocks:

```python
# After the existing Conv/RepC3/ConvNormLayer block:
elif m in (AIFI_SEFFN,):
    c2 = ch[f]
    args = [ch[f], *args]

# For the multi-input DSFD fusion module:
elif m is DSFD:
    c1 = [ch[x] for x in f]
    args = [c1, c2]
```

The `Blocks` and `PCGLU` modules work through the existing `Blocks` handler — no additional registration is needed for `PCGLU` since it is resolved as a `block_type` argument to `Blocks`.

## Step 6: Copy Model Config

```bash
cp models/PSD-RTDETR.yaml /path/to/site-packages/ultralytics/cfg/models/PSD-RTDETR.yaml
```

## Step 7: Train

```python
from ultralytics import RTDETR

model = RTDETR('ultralytics/cfg/models/PSD-RTDETR.yaml')
model.train(data='your_dataset.yaml', epochs=200, batch=8, device='0')
```

## Important Notes

- **AMP must be disabled** (`amp=False` in default.yaml) — RT-DETR produces NaN with mixed-precision training due to `F.grid_sample()`.
- **Default hyperparameters**: `optimizer=AdamW`, `lr0=0.0001`, `warmup_epochs=2000` (iterations, not epochs), `mosaic=0.0`.
- **Windows**: If training hangs during data loading, set `workers=0`.

## Verification

After integration, verify the model builds correctly:

```python
from ultralytics import RTDETR
model = RTDETR('ultralytics/cfg/models/PSD-RTDETR.yaml')
print(model.model)  # Should print the full model architecture
```

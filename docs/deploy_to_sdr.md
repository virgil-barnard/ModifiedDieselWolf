# Deploy to SDR

This guide outlines how to compress a trained DieselWolf model and run it on a low-power device such as a Jetson or Raspberry Pi.

## 1. Prune the checkpoint

Reduce model size by removing 50% of the smallest weights:

```bash
python scripts/prune.py --checkpoint model.ckpt --output pruned.ckpt
```

## 2. Export to ONNX

Convert the pruned weights into the portable ONNX format:

```bash
python scripts/export_onnx.py --checkpoint pruned.ckpt --output model.onnx
```

## 3. Quantise to INT8

Apply post-training quantisation using ONNX Runtime:

```bash
python scripts/quantize_onnx.py --input model.onnx --output model_int8.onnx
```

## 4. Example inference script

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model_int8.onnx")
# `iq.npy` should contain shape (2, N) float32 IQ samples captured from the SDR
input_data = np.load("iq.npy").astype(np.float32)[None]
logits = session.run(None, {"input": input_data})[0]
print("Predicted class:", int(logits.argmax()))
```

Use your preferred SDR library to capture IQ samples and save them to `iq.npy` before running the script.

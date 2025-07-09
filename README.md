# ModifiedDieselWolf

A PyTorch Lightning based research platform for automatic modulation recognition (AMR). The project builds on the original DieselWolf dataset and adds more data augmentations, training utilities and deployment scripts.

## Getting Started

Build the Docker image and start the container:

```bash
docker build -t dieselmod .
docker run --gpus all -it --rm -p 8888:8888 -p 6006:6006 -v $(pwd):/app dieselmod
```

The container launches **JupyterLab** on port `8888` and **TensorBoard** on port `6006`. Navigate to `http://localhost:8888` to open the notebooks and `http://localhost:6006` to monitor training logs.

Inside JupyterLab open a terminal and set the `PYTHONPATH` before running any scripts:

```bash
export PYTHONPATH=$PYTHONPATH:/app
```

For example, you can reproduce the CNN tuning session by running:

```bash
python scripts/tune_cnn.py --max-trials 10
```
TensorBoard includes embedding projections of the best model from each trial.

## Repository Highlights

- **Augmentations:** `dieselwolf/data/augmentations.py` provides `RandomCrop`, `IQSwap` and `RFAugment` which combines random CFO, cropping and IQ swap.
- **Quantisation:** convert checkpoints with `scripts/export_onnx.py` and `scripts/quantize_onnx.py`.
- **Tutorial notebooks:** see the `notebooks/` directory for dataset walkthroughs and training examples.
- Additional transformations for channel simulation live in `dieselwolf/data/TransformsRF.py`.

## Demo

![Confusion matrices](demo_pics/confusion_images.png)
![Parallel coordinates](demo_pics/parallel_coords.png)
![Training curves](demo_pics/scalars.png)
[![Watch the video](demo_pics/confusion_images.png)]
(https://raw.githubusercontent.com/virgil-barnard/ModifiedDieselWolf/main/demo_pics/projector.mp4)

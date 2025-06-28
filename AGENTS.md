# DieselWolf ‑ Development Agents & Roadmap

Welcome!  This checklist tracks every deliverable required to lift DieselWolf from a teaching baseline to a competitive research platform in Automatic Modulation Recognition (AMR).

> **Convention**  
> Tick the `[ ]` box when a task is completed in a PR and reference the hash in parentheses.  
> Sub‑tasks are indented.  Milestones close when *all* nested boxes are ticked.

---

## 1. Repository House‑Keeping
- [x] **Adopt Coding Standards** (`9d6ce92`)
  - [x] Add `ruff` + `black` pre‑commit hooks (`469c913`)
  - [x] Create `CONTRIBUTING.md` and PR template (`469c913`)
- [x] **Environment** (`9d6ce92`)
  - [x] Pin dependencies in `pyproject.toml` (`469c913`)
  - [x] Add `requirements-dev.txt` for docs, lint, tests (`469c913`)

---

- [ ] **Training Loop Migration → PyTorch Lightning**
- [x] Scaffold `LightningModule` for classifier (`9d6ce92`)
 - [x] Remove dependency on reformer_pytorch and refactor Demodulation.ipynb and Modulation_Classifier.ipynb to use the transformer module's ReformerModel instead (`d8dfb69`)
- [x] Move dataset classes to `dieselwolf/data/` (`9d6ce92`)
- [x] Implement Lightning `Trainer` flags (mixed‑precision, callbacks) (`0e8bb61`)
- [x] Add `scripts/train_amr.py` CLI (`0e8bb61`)
 - [x] **CI**: GitHub Action to run a 1‑epoch smoke test on push (`5dcb51d`)

---

## 3. Model‑Architecture Upgrades
- [ ] **Complex‑Valued Core**
  - [ ] Replace `nn.Conv1d`, `BatchNorm1d`, `Linear` with `ComplexConv1d`, etc.
  - [ ] Replace Replace Reformer with a complex‑valued Transformer (CV‑ViT or CC‑MSNet)
  - [ ] Verify weight initialisation & checkpoint I/O
- [ ] **Introduce Radio‑Transformers**
  - [ ] Add **MobileRaT** backbone
  - [ ] Add **NMformer** backbone with noise tokens
  - [ ] Provide YAML configs for both
- [ ] **Hybrid Multitask Heads**
  - [ ] Append SNR‑regression head
  - [ ] Append channel‑parameter (CFO & phase) head
  - [ ] Balance multi‑loss weights via grid search

---

## 4. Representation Learning
- [ ] **Self‑Supervised Pre‑Training (SSL)**
  - [ ] Implement MoCo‑v3 queue & momentum encoder
  - [ ] Create `RFAugment` with: random CFO, time cropping, IQ swap
  - [ ] Pre‑train on synthetic + RadioML 2016/2018
- [ ] **Curriculum over SNR**
  - [ ] Schedule: start at +20 dB, lower by 5 dB every plateau
  - [ ] Add callback to adjust sampling weights

---

## 5. Channel & Data‑Generation Enhancements
- [x] **DopplerShift Transform** (`5dcb51d`)
  - [x] Fractional‑resample via `torchaudio.functional.resample` (`5dcb51d`)
  - [x] Unit tests on sample delay accuracy (`5dcb51d`)
- [ ] **Extended Fading Profiles**
  - [ ] TDL Rayleigh
  - [ ] Rician K‑factor sweep
  - [ ] Nakagami‑m option
- [ ] **Interference Simulation**
  - [ ] Add co‑channel QPSK interferer mix‑in
  - [ ] Label jammer SNR for possible auxiliary training

---

## 6. Training & Optimisation
- [ ] **Optimisers & Schedulers**
  - [ ] Switch default to Lookahead(AdamW)
  - [ ] Plug cosine LR decay with warm‑up
- [ ] **Mixed Precision + Gradient Accumulation**
  - [ ] Enable AMP in Trainer
  - [ ] Tune `accumulate_grad_batches` for batch‑equivalent training
- [ ] **Checkpoint & Resume**
  - [ ] Save EMA weights
  - [ ] Resume SSL → fine‑tune seamlessly

---

## 7. Model Compression & Deployment
- [ ] **Pruning**
  - [ ] Global magnitude prune to 50 %
  - [ ] Re‑fine‑tune 3 epochs to regain accuracy
- [ ] **INT8 Quantisation**
  - [ ] Export `.onnx` model
  - [ ] Quantise with ONNX‑Runtime PTQ
  - [ ] Benchmark on Jetson‑Nano / Raspberry Pi
- [ ] **Documentation**
  - [ ] Write “Deploy to SDR” guide with example Python script

---

## 8. Evaluation & Benchmarks
- [ ] **Datasets**
  - [ ] Integrate RadioML 2016, 2018, and DeepSig RML22 loaders
- [ ] **Metrics**
  - [ ] Accuracy vs SNR curve
  - [ ] Confusion matrix at 0 dB
  - [ ] Inference latency on CPU & GPU
- [ ] **Reproducibility**
  - [ ] Dockerfile with exact CUDA/cuDNN
  - [ ] GitHub Action: regenerate benchmark plots on push to `main`

---

## 9. Project Management
- [ ] **GitHub Projects Board**
  - [ ] Map these check‑boxes to issues
  - [ ] Automate “Done” column via PR merge events
- [ ] **Milestone Labels**
  - [ ] `v0.2-ssl`, `v0.3-transformer`, `v1.0-release`

---

Good luck — and please link every PR back to the relevant tick box!
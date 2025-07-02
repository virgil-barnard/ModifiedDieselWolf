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
- [x] **Training Loop Migration → PyTorch Lightning** (`00a5384`)
- [x] Scaffold `LightningModule` for classifier (`9d6ce92`)
 - [x] Remove dependency on reformer_pytorch and refactor Demodulation.ipynb and Modulation_Classifier.ipynb to use the transformer module's ReformerModel instead (`d8dfb69`)
- [x] Move dataset classes to `dieselwolf/data/` (`9d6ce92`)
- [x] Implement Lightning `Trainer` flags (mixed‑precision, callbacks) (`0e8bb61`)
- [x] Add `scripts/train_amr.py` CLI (`0e8bb61`)
 - [x] **CI**: GitHub Action to run a 1‑epoch smoke test on push (`5dcb51d`)

---

## 3. Model‑Architecture Upgrades
- [x] **Complex‑Valued Core** (`c84d441`)
  - [x] Replace `nn.Conv1d`, `BatchNorm1d`, `Linear` with `ComplexConv1d`, etc. (`2cd4d66`)
  - [x] Replace Replace Reformer with a complex‑valued Transformer (CV‑ViT or CC‑MSNet) (`1cfef0b`)
  - [x] Verify weight initialisation & checkpoint I/O (`6d28f30`)
- [x] **Introduce Radio‑Transformers** (`df5681a`)
  - [x] Add **MobileRaT** backbone (`c84d441`)
  - [x] Add **NMformer** backbone with noise tokens (`78d6a02`)
  - [x] Provide YAML configs for both (`78d6a02`)
- [x] **Hybrid Multitask Heads** (`6c46bf6`)
  - [x] Append SNR‑regression head (`df5681a`)
  - [x] Append channel‑parameter (CFO & phase) head (`02e77b3`)
  - [x] Balance multi‑loss weights via grid search (`6c46bf6`)

---

## 4. Representation Learning
- [ ] **Self‑Supervised Pre‑Training (SSL)**
  - [x] Implement MoCo‑v3 queue & momentum encoder (`335881d`)
  - [x] Create `RFAugment` with: random CFO, time cropping, IQ swap (`335881d`)
  - [x] Pre‑train on synthetic + RadioML 2016/2018 (`a8e39db`) 
- [ ] **Curriculum over SNR**
  - [x] Schedule: start at +20 dB, lower by 5 dB every plateau (`a8e39db`) 
  - [x] Add callback to adjust sampling weights (`a8e39db`) 

---

## 5. Channel & Data‑Generation Enhancements
- [x] **DopplerShift Transform** (`5dcb51d`)
  - [x] Fractional‑resample via `torchaudio.functional.resample` (`5dcb51d`)
  - [x] Unit tests on sample delay accuracy (`5dcb51d`)
- [x] **Extended Fading Profiles** (`83b3629`)
  - [x] TDL Rayleigh (`7f76c44`)
  - [x] Rician K‑factor sweep (`83b3629`)
  - [x] Nakagami‑m option (`83b3629`)
- [x] **Interference Simulation** (`098684d`)
  - [x] Add co‑channel QPSK interferer mix‑in (`098684d`)
  - [x] Label jammer SNR for possible auxiliary training (`098684d`)

---

## 6. Training & Optimisation
- [x] **Optimisers & Schedulers** (`00a5384`)
  - [x] Switch default to Lookahead(AdamW) (`00a5384`)
  - [x] Plug cosine LR decay with warm‑up (`00a5384`)
- [x] **Mixed Precision + Gradient Accumulation** (`00a5384`)
  - [x] Enable AMP in Trainer (`00a5384`)
  - [x] Tune `accumulate_grad_batches` for batch‑equivalent training (`00a5384`)
- [ ] **Checkpoint & Resume**
  - [x] Save EMA weights (`ee33620`)
  - [x] Resume SSL → fine‑tune seamlessly (`a8e39db`) 

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
- [x] **Datasets**
  - [x] Integrate RadioML 2016, 2018, and DeepSig RML22 loaders (`335881d`)
- [x] **Metrics**
  - [x] Accuracy vs SNR curve (`335881d`)
  - [x] Confusion matrix at 0 dB (`335881d`)
  - [x] Inference latency on CPU & GPU (`335881d`)
- [x] **Reproducibility**
  - [x] Dockerfile with exact CUDA/cuDNN (`a8e39db`)
  - [x] GitHub Action: regenerate benchmark plots on push to `main` (`f4cc76e`)

---

## 9. Project Management
- [ ] **GitHub Projects Board**
  - [ ] Map these check‑boxes to issues
  - [ ] Automate “Done” column via PR merge events
- [ ] **Milestone Labels**
  - [ ] `v0.2-ssl`, `v0.3-transformer`, `v1.0-release`

---

Good luck — and please link every PR back to the relevant tick box!
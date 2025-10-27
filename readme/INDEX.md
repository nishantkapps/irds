# Documentation Index

Complete documentation for the IRDS Gesture Recognition Project.

## 📖 Documentation Guide

### 🚀 Getting Started (Start Here!)

**[QUICK_START.md](QUICK_START.md)**
- Quick commands to run experiments
- Simple examples for HPC deployment
- Troubleshooting tips
- **Recommended for first-time users**

---

### 🔬 Experiments & Training

**[README_EXPERIMENTS.md](README_EXPERIMENTS.md)**
- Complete guide to multi-model experiments
- Detailed training instructions
- Configuration file structure
- Running experiments on HPC
- SLURM batch job examples
- **Read this for understanding the experiment framework**

**[MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md)**
- Technical overview of 6 model architectures
- Model parameter counts and specifications
- Configuration system details
- Output file structure
- **For understanding model architecture choices**

---

### 🧹 Cleanup Documentation

**[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)**
- Training directory cleanup details
- Files removed from `model/train/`
- Removal of `--skip_gpu_test` flag
- Updated command examples
- **Background info on code cleanup**

**[MODEL_CLEANUP_SUMMARY.md](MODEL_CLEANUP_SUMMARY.md)**
- Model directory cleanup details  
- Obsolete files removed (~45MB freed)
- Before/after directory structure
- New output file organization
- **Background info on model cleanup**

---

### 🎥 Video Inference

**[VIDEO_PIPELINE_GUIDE.md](VIDEO_PIPELINE_GUIDE.md)**
- Video-to-gesture inference pipeline
- Using trained models for video prediction
- MediaPipe skeleton extraction
- Testing utilities
- **For running inference on videos**

---

### 🖥️ HPC Deployment

**[HPC_FILE_LIST.md](HPC_FILE_LIST.md)**
- Files needed for HPC deployment
- Rsync commands
- Directory structure for HPC
- **Quick reference for deploying to HPC**

---

## 📋 Quick Navigation

| I want to... | Read this document |
|--------------|-------------------|
| Start training quickly | [QUICK_START.md](QUICK_START.md) |
| Understand the experiments | [README_EXPERIMENTS.md](README_EXPERIMENTS.md) |
| Learn about model architectures | [MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md) |
| Run inference on videos | [VIDEO_PIPELINE_GUIDE.md](VIDEO_PIPELINE_GUIDE.md) |
| Deploy to HPC | [HPC_FILE_LIST.md](HPC_FILE_LIST.md) |
| Understand code changes | [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) + [MODEL_CLEANUP_SUMMARY.md](MODEL_CLEANUP_SUMMARY.md) |

---

## 🎯 Recommended Reading Order

### For New Users
1. [QUICK_START.md](QUICK_START.md) - Get started immediately
2. [README_EXPERIMENTS.md](README_EXPERIMENTS.md) - Understand experiments
3. [MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md) - Learn about models

### For HPC Deployment
1. [QUICK_START.md](QUICK_START.md) - Commands overview
2. [HPC_FILE_LIST.md](HPC_FILE_LIST.md) - Files to copy
3. [README_EXPERIMENTS.md](README_EXPERIMENTS.md) - Running on HPC

### For Development
1. [MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md) - Architecture details
2. [README_EXPERIMENTS.md](README_EXPERIMENTS.md) - Experiment framework
3. [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - Recent changes

---

## 📁 All Documentation Files

```
readme/
├── INDEX.md                          # This file
├── QUICK_START.md                    # Quick reference (5 min read)
├── README_EXPERIMENTS.md             # Complete experiments guide (15 min read)
├── MULTI_MODEL_SETUP.md              # Model architecture details (10 min read)
├── VIDEO_PIPELINE_GUIDE.md           # Video inference guide (10 min read)
├── HPC_FILE_LIST.md                  # HPC deployment reference (5 min read)
├── CLEANUP_SUMMARY.md                # Training cleanup details (5 min read)
└── MODEL_CLEANUP_SUMMARY.md          # Model cleanup details (5 min read)
```

---

**Tip**: Start with [QUICK_START.md](QUICK_START.md) if you just want to run experiments!


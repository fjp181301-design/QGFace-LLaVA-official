# QGFace-LLaVA

Official implementation of **QGFace-LLaVA: Quality-Aware Controlled Fusion of Structured Side Information for Face Analysis under Imperfect Metadata**.

## Overview

QGFace-LLaVA is a quality-aware controlled fusion framework for multimodal large language model (MLLM)-based face analysis under imperfect metadata.

The method focuses on a key problem in metadata-aware face analysis: structured side information such as age, gender, and confidence cues is not always reliable or universally beneficial. Such metadata may help prediction when it is accurate and task-relevant, but it may also introduce misleading bias when it is noisy, missing, or mismatched with the input image.

QGFace-LLaVA preserves visual dominance while regulating metadata influence according to task context and estimated metadata quality.

## Key Idea

The main idea of QGFace-LLaVA is to treat structured metadata as a conditional auxiliary prior rather than a dominant input source.

The framework contains three key designs:

- dominant visual representation construction;
- task-aware metadata quality estimation;
- quality-aware controlled fusion before LLM-based task prediction.

## Datasets

Experiments are conducted on three representative face analysis benchmarks:

| Dataset | Task | Metric |
|---|---|---|
| FER2013 | Facial expression recognition | Accuracy |
| CelebA-40 | Facial attribute recognition | mAcc |
| UTKFace | Age estimation | MAE |

## Metadata Settings

The method is evaluated under both clean and imperfect metadata conditions:

| Setting | Description |
|---|---|
| Clean | Metadata are used without corruption |
| Noisy60 | Metadata are partially corrupted |
| Missing | Metadata are incomplete or unavailable |
| Shuffled | Metadata are mismatched across samples |

## Compared Methods

The experiments compare four metadata usage strategies:

| Method | Description |
|---|---|
| Base | Visual-only prediction |
| Naive Fusion | Direct metadata injection |
| Gate Only | Metadata fusion with adaptive gating |
| QGFace-LLaVA | Quality-aware controlled metadata fusion |

## Main Finding

The results show that structured side information is not universally beneficial in face analysis. Its practical value depends on task relevance, metadata quality, and fusion strategy.

QGFace-LLaVA provides a more reliable way to use metadata by suppressing misleading auxiliary cues while retaining useful information when metadata are trustworthy and task-relevant.

## Repository Status

This repository provides a minimal public release for QGFace-LLaVA, including environment requirements, dataset organization instructions, configuration files, and basic training/evaluation entry scripts.

The original datasets and experimental result files are not redistributed in this repository due to dataset license restrictions and file-size considerations.

## Installation

```bash
conda create -n qgface-llava python=3.10 -y
conda activate qgface-llava
pip install -r requirements.txt
```

## Data Preparation

Please download the public datasets from their official sources and organize them as follows:

```text
data/
├── FER2013/
├── CelebA/
└── UTKFace/
```

Due to dataset license restrictions, the original datasets are not redistributed in this repository.

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{feng2026qgfacellava,
  title={QGFace-LLaVA: Quality-Aware Controlled Fusion of Structured Side Information for Face Analysis under Imperfect Metadata},
  author={Feng, Jinping and Xu, Nan and Li, Xi and Fu, Zhongtao and Xiao, Zhenhua and Huang, Zhenghua},
  journal={Scientific Reports},
  year={2026}
}
```

## License

This project is released under the MIT License.

## Contact

For questions about this repository, please contact:

Xi Li  
Email: lixi@wit.edu.cn

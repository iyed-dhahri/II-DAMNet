# Bridging the Nocturnal Data Gap via Cycle-Consistent Domain Translation for Illumination-Invariant Semantic Segmentation in UAV Disaster Imagery

Official repository for **II-DAMNet** (Illumination-Invariant Damage Assessment Model). [cite_start]This framework addresses the scarcity of annotated nighttime aerial imagery by using structure-preserving day-to-nocturnal domain translation to enable reliable, round-the-clock automatic post-disaster evaluation[cite: 6, 11].

---

## 📌 Overview

[cite_start]Drones (UAVs) equipped with computer vision systems are crucial for minimizing response delays during natural disasters[cite: 5, 17]. [cite_start]However, their performance drops drastically at night due to a severe lack of annotated nocturnal data[cite: 5]. 

[cite_start]This project bridges that gap by implementing two simulated datasets (**Twilight-style** and **Infrared-style**) derived from the daytime RescueNet dataset[cite: 6, 43]. [cite_start]By leveraging translation methods with strict cycle-consistency (physics-based **KinD2** and heuristic **Synt1** filters), we manipulate illumination maps and color temperatures without destroying the underlying geometric structures or invalidating original daytime segmentation masks[cite: 38, 40, 42].


### Key Highlights
* [cite_start]**Zero-Annotation Overhead:** Automatically expands existing daytime datasets into nocturnal domains without manual re-labeling[cite: 6, 7].
* [cite_start]**Robust Under Extreme Shift:** Boosts mean Intersection over Union (mIoU) on challenging simulated infrared domains from **24.69%** (baseline PSPNet) to **60.40%**[cite: 10].
* [cite_start]**Consistent Cross-Domain Performance:** Maintains stable semantic accuracy across day, twilight, and infrared simulated scenarios (60.40%–66.40% mIoU)[cite: 10].

---

## 🏗️ Framework Architecture

The workflow consists of two main pillars:
1. [cite_start]**Cycle-Consistent Generator Selection:** Evaluating translation candidates via a pre-trained daytime segmentation model to verify spatial integrity[cite: 54, 57].
2. [cite_start]**Hybrid Dataset Training:** Training the **II-DAMNet** architecture (built on a ResNet-101 backbone with a Pyramid Pooling Module) using a mixed-domain training configuration[cite: 65, 96, 166].

---

## 📊 Performance Summary

### Quantitative Results (IoU % / mIoU)
Compared against the state-of-the-art PSPNet baseline on the grouped critical classes (*Intact Building*, *Damaged Building*, *Blocked Road*, and *Background*)[cite: 10, 70, 71]:

| Model | Original (Day) | Infrared-style | Twilight-style | Mixed Test Set |
| :--- | :---: | :---: | :---: | :---: |
| **PSPNet (Baseline)** | **74.41%** | 24.69% | 49.64% | 50.58% |
| **II-DAMNet (Ours)** | 66.40% | **60.40%** | **62.91%** | **61.71%** |

---

## 📂 Repository Structure
*(Codebase integration is currently in progress)*
```text
├── datasets/             # Data structure and simulation scripts
├── models/               # II-DAMNet and backbone definitions
├── core/                 # Training and evaluation logic
├── weights/              # Pre-trained model checkpoints placeholder
├── requirements.txt      # Environment dependencies
└── README.md

% Placeholder for your future citation details
@article{dhahri2026bridging,
  title={Bridging the Nocturnal Data Gap via Cycle-Consistent Domain Translation for Illumination-Invariant Semantic Segmentation in UAV Disaster Imagery},
  author={Dhahri, Iyed and Golabi, Mahmoud and Hammoudi, Karim and Idoumghar, Lhassane},
  journal={AVSS},
  year={2026}
}

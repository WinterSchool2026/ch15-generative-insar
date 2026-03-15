# Generative Models for Interferometric Synthetic Aperture Radar

> Interferometric Synthetic Aperture Radar (InSAR) is a powerful remote sensing modality that
can provide highly accurate information on ground displacement by examining the phase
differences between Synthetic Aperture Radar acquisitions captured at different times on the same location. InSAR has become indispensable for monitoring earthquakes, volcanic
activity, landslides, subsidence, and infrastructure stability. However, the limited frequency of
such geophysical events results in scarce labeled datasets, hindering the application of deep
learning methods in this high-impact domain. This challenge aims to address this problem
through synthetic InSAR generation.
The objectives of this challenge can be summarized as follows:
a) Model Development: Design methods that can generate synthetic samples
conditioned on predefined concepts or textual description of InSAR data;
b) Evaluation Protocol: Construct an evaluation protocol that assesses the quality of the
generated InSAR, taking into consideration both the generated image quality as well
as the underlying physics;
c) Practical utility assessment: Test whether synthetically generated InSAR can be used
effectively as a training dataset for supervised learning.


---

 
## 📋 Table of Contents
- [Challenge Overview](#challenge-overview)
- [Scope](#scope)
- [Goal](#goal)
- [Interesting Reads](#interesting-reads)
- [Where to Start](#where-to-start)
- [Repository Structure](#repository-structure)
- [Contact](#contact)

---
 
<h2 id="challenge-overview">📡 Challenge Overview</h2>

This challenge focuses on **generative models for Interferometric Synthetic Aperture Radar (InSAR)**, a domain where standard computer vision tools fall short, data is scarce, and the definition of "good" generation is highly ambiguous.

Unlike natural image generation, InSAR images carry physical meaning encoded in phase, coherence, and geometric structure. Generating them well, and evaluating whether you've done so, requires rethinking assumptions borrowed from the natural image world.

---

<h2 id="scope">🧭 Scope</h2>

The challenge revolves around two interrelated axes:

**a) Synthetic InSAR Generation** — generating realistic InSAR imagery under highly skewed class distributions with hierarchical label categories.

**b) Evaluation Framework Design** — formulating novel metrics to assess generated InSAR across three dimensions: realism, faithfulness to conditioning, and physical feasibility. The resulting framework will move away from (or extend) standard natural image metrics (FID etc.) and ground them on the nature of InSAR imagery and feasibility.

---

<h2 id="goal">🎯 Goal</h2>

 Think of this challenge as an open invitation to follow your curiosity and intuitions into modelling, evaluation methodology, or wherever the problem takes you. 
 
  Don't be discouraged by the scale of the challenge. Both axes are genuinely hard and genuinely interesting, spanning a wide range of open problems and covering everything from evaluation gaps in the literature to data and resource constraints. You don't need to solve everything. Sometimes a baby step lands exactly where it's needed.

---

<h2 id="interesting-reads">📚 Interesting Reads</h2>

### Generative Modelling

| Paper | Notes |
|---|---|
| [ProjectedGANs](https://arxiv.org/abs/2111.01007) · [GitHub](https://github.com/autonomousvision/projected-gan) | Robust image generation on limited data — proven to work on limited InSAR data |
| [Image-to-Image Translation (Pix2Pix)](https://arxiv.org/pdf/1611.07004) | Imate to Image generation. Can serve as an inspiration on how to use image-like data (e.g., DEM, coherence, atmospheric variables) as conditioning for synthetic InSAR generation. Can ideas be blended with ProjectedGANs? |
| [Image-to-Many Images. (BicycleGAN)](https://arxiv.org/abs/1711.11586)| Pix2Pix follow-up with random noise
|[Class conditioning with limited data](https://arxiv.org/pdf/2201.06578) | Training strategy for stable and easier class conditioning in low-data regimes
| [Introduction to Diffusion Models](https://arxiv.org/abs/2506.02070) | A brief introduction into diffusion models. Feel free to investigate methodologies. E.g Diffusion with concatenated auxiliary data (DEM etc.) at every step t|
| [Palette](https://arxiv.org/pdf/2111.05826) | Image-To-Image conditional diffusion models via concatenation|
|[SR3](https://arxiv.org/pdf/2104.07636) | Image Super-Resolution conditioned on low-resolution input. Palette built on this idea.|
| [Lucidrains Diffusion Library](https://codeberg.org/lucidrains/denoising-diffusion-pytorch) | Useful library for training diffusion models. |

### Assessing Synthetic Image Quality

| Paper | Notes |
|---|---|
| [The Role of ImageNet Classes in FID](https://arxiv.org/abs/2203.06026) | Effect of ImageNet-pretrained feature space on FID. How meaningful is this setup for out-of-domain data like InSAR? |
| [Rethinking FID](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf) | Limitations of FID and the case for CMMD as a replacement |

### Discussion on Natural Image Quality Metrics

| Paper | Notes |
|---|---|
| [Image Evaluation Study](https://arxiv.org/pdf/2304.01999) | Empirical investigation into image evaluation metrics. |
| [How Good Are Image Quality Measures?](https://arxiv.org/pdf/2201.13019) | Assessment of existing image quality metrics. |
| [A Perspective on Image Realism](https://raw.githubusercontent.com/mlresearch/v235/main/assets/theis24a/theis24a.pdf) | Position paper on how "image realism" should and shouldn't be defined. |

### Feasibility of Generative Models

| Paper | Notes |
|---|---|
| [Do Generative Video Models Understand Physical Principles?](https://arxiv.org/abs/2501.09038) | Benchmarks physical feasibility of video generative models and proposes a unique evaluation framework to assess physical understanding. Can be a key inspiration for the evaluation axis of this challenge |
| [Generative Physical AI in Vision](https://arxiv.org/pdf/2501.10928) | Survey on physics-aware image/video/3d-4d generation and evaluation. Can serve as inspiration |

### InSAR Basics

| Resource| Notes|
|---|---|
|[InSAR Resources](https://www.esa.int/About_Us/ESA_Publications/InSAR_Principles_Guidelines_for_SAR_Interferometry_Processing_and_Interpretation_br_ESA_TM-19)| ESA notes for InSAR processing|

### Challenge data

|Paper| Notes|
|---|---|
|[Thalia](https://arxiv.org/abs/2505.17782)| Thalia, a global multi-modal dataset for volcanic unrest monitoring. Thalia integrates InSAR, coherence, DEM, and atmospheric data across 38 spatiotemporal datacubes spanning 7 years of volcanic activity.|
---

<h2 id="where-to-start">💡 Where to Start</h2>

If you're not sure where to jump in, here are a few natural entry points:

- **InSAR generation:** Start with ProjectedGANs and try to generate meaningful InSAR. **Then ask: How can I control what to generate?** Condition on class. Attempt to generate meaningful InSAR with and without ground deformation. Moving further. Can I control the topography? Condition on DEM. Can I introduce atmospheric contributions? Enter atmospheric variables.
- **Evaluation of synthetic InSAR:** Understand existing metrics and their limitations. Think what principles should InSAR data follow and how can we ground them on plausibility. Does simple FID from Imagenet-pretrained InceptionV3 work? Can we create an InSAR based feature space for evaluation? Can we move further to domain-specifc evaluation? Read the feasibility papers and **ask: what makes an InSAR image physically inadmissible, and can we detect that automatically?**

There is no single right answer, and partial contributions such as a new metric, a negative result, or a cool baseline, are all valuable.

<p align="center">
  <img src="https://media1.tenor.com/m/d2NYSXokaK4AAAAC/pikachu-cheer-dance.gif" width="400">
</p>

---

<h2 id="repository-structure">🗂️ Repository Structure</h2>

```
.
├── ThaliaDemo.ipynb            # Notebook demonstrating how to download, visualize, explore, and iterate over the Thalia dataset.
└── data_loading_utilities.py   # Data loading utilities. Used in the notebook.
```

---

<h2 id="contact">📬 Contact</h2>

For questions, open an issue or reach out to the challenge organiser.

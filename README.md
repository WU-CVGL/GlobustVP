<p align="center">
  <h1 align="center"> <ins>GlobustVP</ins> 💥<br>Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World</h1>
  <h3 align="center">CVPR 2025 Award Candidate & Oral</h3>
  <p align="center">
    <span class="author-block">
      <a href="https://bangyan101.github.io/">Bangyan Liao</a><sup>*</sup>
      ·
      <a href="https://ericzzj1989.github.io/">Zhenjun Zhao</a><sup>*</sup>
      ·
      <a href="https://sites.google.com/view/haoangli/homepage">Haoang Li</a>
      ·
      <a href="https://sites.google.com/view/zhouyi-joey/home">Yi Zhou</a>
      ·
      <a href="">Yingping Zeng</a>
      ·
      <a href="">Hao Li</a>
      ·
      <a href="https://ethliup.github.io/">Peidong Liu</a><sup>†</sup>
    </span>&nbsp;&nbsp;&nbsp;&nbsp;
  </p>

  <p align="center">
    <sup>*</sup> equal contribution, <sup>†</sup> corresponding author
  </p>

  <div align="center">

  [![arXiv](https://img.shields.io/badge/arXiv-2505.04788-b31b1b.svg)](https://arxiv.org/abs/2505.04788)
  [![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

  </div>
</p>

Official implementation of the CVPR 2025 Award Candidate & Oral paper:
**"Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World"**

<p align="center">
  <a href="https://arxiv.org/abs/2505.04788">
    <img src="./media/teaser.jpg" alt="Logo" width=80%>
  </a>
  <br>
  <em>A globally optimal and outlier-robust method for vanishing point (VP) estimation in a Manhattan world, using convex relaxation techniques.</em>
</p>

## 🔍 Overview

**GlobustVP** is a globally optimal and outlier-robust method for vanishing point (VP) estimation under the Manhattan World assumption. For the first time, we introduce **convex relaxation techniques** into VP estimation by reformulating the problem as a quadratically constrained quadratic program (QCQP), and then relaxing it into a convex semidefinite program (SDP). This approach avoids the limitations of local minima in iterative solvers and scales to realistic noise and outlier settings.

## ✨ Highlights

- **Global optimality** without initialization
- **Robust to outliers** up to 70%
- **Efficient** runtime (~50ms/image)
- **State-of-the-art** performance on YUD and SU3 datasets
- **No deep learning or training** required

## 📦 Installation

**Dependencies**:
- Python ≥ 3.8
- `numpy`, `matplotlib`, `scipy`
- `cvxpy` with an SDP solver (e.g., [MOSEK](https://www.mosek.com/), `SCS`, or `CVXOPT`)


Install via pip:

```bash
git clone https://github.com/ericzzj1989/GlobustVP.git && cd GlobustVP
python -m pip install -e .
```

## 🚀 Getting Started

### 1. Prepare config file

Create a JSON configuration file (e.g., default file `config/param.json`):

```json
{
  "line_num": 120,
  "iteration": 100,
  "outlier_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
  "sample_line_num": 6,
  "vanishing_point_num": 3,
  "endpoint_noise": 1.0,
  "line_length": 4.0,
  "gt_vp_noise": 0.001,
  "c": 0.03,
  "K": [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
}
```

### 2. Run experiment

```bash
python -m globustvp.run_experiments --config globustvp/config/param.json
```

### 3. View results

- Results saved in `results/experiment_results.json`
- Figures saved in `figures/`
- Boxplots of precision, recall, and F1-score are automatically generated

## 🗃️ Code Structure

```graphql
├── run_experiments.py              # Main pipeline
├── solver/
│   ├── core.py                     # GlobusVP solver
│   ├── sdp_solver.py               # SDR formulation and SDP solver
│   └── solver_utils.py             # Supporting methods for solving
├── utils/
│   ├── data.py                     # Synthetic data generation
│   ├── geometry.py                 # Geometric projection and normalization
│   ├── io.py                       # Result saving and loading
│   ├── metrics.py                  # Evaluation metrics
│   └── plot.py                     # Boxplot and visualization
```

## 📊 Results

### Synthetic Data (F1-Score vs. Outlier Ratio)
![boxplot](media/synthetic_outlier_f1.jpg)

### Real-World Results (YUD)
| Method       | AA@3° | AA@5° | AA@10° |
|--------------|-------|-------|--------|
| J-Linkage [44] | 57.7 | 69.3  | 80.5 |
| Quasi-VP [31] | 57.8 | 72.5 | 84.3 |
| NeurVPS [51] | 52.2 | 64.2  | 78.1   |
| GlobustVP 🏆 | **67.6** | **87.3** | **96.1** |

## 📁 Datasets

- [YUD (York Urban Database)](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/)
- [SU3 (SceneCity Urban 3D Wireframe Dataset)](https://github.com/zhou13/shapeunity)
- [NYU-VP](https://github.com/fkluger/nyu_vp)

## 📝 Citation

If you use this code or paper, please cite:

```bibtex
@inproceedings{liao2025globustvp,
  title={Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World},
  author={Liao, Bangyan and Zhao, Zhenjun and Li, Haoang and Zhou, Yi and Zeng, Yingping and Li, Hao and Liu, Peidong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## 📬 Contact

For questions or feedback, feel free to contact:

- [Bangyan Liao](mailto:liaobangyan@westlake.edu.cn) 
- [Zhenjun Zhao](mailto:ericzzj89@gmail.com)
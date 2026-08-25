# Adaptive Quantization Observer Evaluation Framework

This repository contains the source code and experimental framework for the research on **Adaptive Quantization Observers (AQO)** for Hoeffding Tree Regressors. The project utilizes a **custom fork** of the [River](https://riverml.xyz/) library to implement and evaluate novel splitting criteria against established baselines in concept drift scenarios.

## 📋 Overview

The main goal of this framework is to evaluate the impact of different kernel density estimation methods (Triangular, Epanechnikov, Smooth) within the Quantization Observer splitter. The experiments measure performance across synthetic datasets (with controlled drift) and real-world data streams.

### Key Features

* **Prequential Evaluation:** Rigorous testing using parallel processing.
* **Novel Splitters:** Implementation and testing of `HTR-AQO` (Adaptive Quantization Observer).
* **Drift Simulation:** Comprehensive synthetic data generation (Friedman, Hyperplane, RBF) with abrupt and gradual drifts.
* **Visualization:** Automated plotting scripts for CD diagrams and performance analysis over time.

## 🛠️ Installation

To reproduce the experiments, it is recommended to use a virtual environment. **Crucially, this project requires a specific branch of the River library fork.**

```bash
# 1. Clone the repository
git clone https://github.com/caiogimenes/river-evaluate.git
cd river-evaluate

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install experiment dependencies
pip install -r requirements.txt

# 3b. Optional: Jupyter stack for the analysis notebooks
pip install -r requirements-dev.txt

# 4. Install the custom River fork (Specific branch 'feat/adaptive-qo' is required)
pip install git+https://github.com/caiogimenes/river.git@feat/adaptive-qo

```

**Main Dependencies:**

* Python 3.10+
* `river` (Fork: [caiogimenes/river](https://github.com/caiogimenes/river), Branch: `feat/adaptive-qo`)
* `numpy`
* `pandas`
* `matplotlib` / `seaborn`
* `joblib`

## 📂 Project Structure

```text
river-evaluate/
├── configs/              # Experiment YAML (instances, seed, datasets, output)
├── experiments/          # CLI to run the prequential evaluation
├── logs/                 # Raw experiment results (.pkl + .json manifest)
├── notebooks/            # Post-hoc analysis of logs
├── output/               # Generated plots and diagrams
├── src/
│   ├── data/             # Data generators and adapters (Synthetic & Real)
│   ├── evaluation/       # Prequential loop + RunnerLog
│   ├── experiment/       # YAML config, dataset assembly, pickle/manifest
│   ├── models/           # Definition of Regressors and Splitters
│   ├── plot/             # Visualization utilities
│   ├── stats/            # Friedman / Nemenyi / ranking
│   ├── logging_setup.py
│   ├── paths.py          # REPO_ROOT and resolve_repo_path
│   └── utils.py          # Notebook-compatible re-exports
├── run_experiment.py     # Thin wrapper around the CLI (same defaults)
├── requirements.txt      # Experiment dependencies
└── requirements-dev.txt  # Jupyter / notebook stack

```

## 🚀 Usage

To run the full experimental suite, execute the main script (defaults live in `configs/experiment.yaml`):

```bash
python run_experiment.py
# equivalent:
python experiments/run_prequential.py --config configs/experiment.yaml
```

Useful overrides:

```bash
python experiments/run_prequential.py --instances 1000 --seed 42 --n-jobs 4 --output logs/debug.pkl
```

*Note: the default is still 1,000,000 instances per dataset. A master `seed` (default 42) is applied before sampling synthetic stream parameters so later runs can be reproduced. Pickles written before this refactor were generated without a master seed.*

After a run, results are written to `logs/gradual.pkl` plus a JSON sidecar (`logs/gradual.json`) with seed, instance count, models, and timestamp.

## 🧪 Experimental Setup

### Models Evaluated

The experiments compare the following variations of Hoeffding Tree Regressors (HTR):

1. **Baselines:**
* `HATR`: Hoeffding Adaptive Tree Regressor (Standard implementation).
* `HTR-QO-0.25`: HTR with Quantization Observer (radius=0.25).
* `HTR-QO-0.5`: HTR with Quantization Observer (radius=0.5).


2. **Proposed Methods (Adaptive QO):**
* `HTR-AQO-Triangular`: Adaptive QO with Triangular kernel.
* `HTR-AQO-Epanechnikov`: Adaptive QO with Epanechnikov kernel.
* `HTR-AQO-Smooth`: Adaptive QO with Smooth kernel.



### Datasets

The framework utilizes a diverse set of data streams:

* **Synthetic:** Friedman (Gradual/Abrupt Drift), Hyperplane, RandomRBF.
* **Real-world:** Bikes, Elec2, CoverType.

## 📊 Results & Visualization

After running the experiments, logs are saved in the `logs/` directory. You can use the notebooks in `notebooks/` (start with `log_analysis.ipynb`) or the scripts in `src/plot/` to generate:

* Performance over time plots.
* Critical Difference (CD) diagrams.
* Resource usage analysis (Memory/Time).

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{DIAS, C.G.,
  title={Adaptive Quantization Observers for Online Regression Trees},
  author={DIAS, C.G.]},
  journal={},
  year={2026}
}

```

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

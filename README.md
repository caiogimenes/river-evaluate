# Adaptive Quantization Observer Evaluation Framework

This repository contains the source code and experimental framework for the research on **Adaptive Quantization Observers (AQO)** for Hoeffding Tree Regressors. The project utilizes the [River](https://riverml.xyz/) library for online machine learning to evaluate the performance of novel splitting criteria against established baselines in concept drift scenarios.

## 📋 Overview

The main goal of this framework is to evaluate the impact of different kernel density estimation methods (Triangular, Epanechnikov, Smooth) within the Quantization Observer splitter. The experiments measure performance across synthetic datasets (with controlled drift) and real-world data streams.

### Key Features

* **Prequential Evaluation:** rigorous testing using parallel processing.
* **Novel Splitters:** Implementation and testing of `HTR-AQO` (Adaptive Quantization Observer).
* **Drift Simulation:** Comprehensive synthetic data generation (Friedman, Hyperplane, RBF) with abrupt and gradual drifts.
* **Visualization:** Automated plotting scripts for CD diagrams and performance analysis over time.

## 🛠️ Installation

To reproduce the experiments, it is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/caiogimenes/river-evaluate.git
cd river-evaluate

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

```

**Main Dependencies:**

* Python 3.10+
* `river`
* `numpy`
* `pandas`
* `matplotlib` / `seaborn`
* `joblib`

## 📂 Project Structure

```text
river-evaluate/
├── logs/                 # Stores raw experiment results (.pkl files)
├── output/               # Generated plots and diagrams
├── src/
│   ├── data/             # Data generators and adapters (Synthetic & Real)
│   ├── models/           # Definition of Regressors and Splitters
│   ├── plot/             # Visualization utilities
│   ├── stats/            # Statistical tests (Friedman, Nemenyi)
│   └── utils.py          # Evaluation loops
├── run_experiment.py     # Main entry point for execution
├── log_analysis.ipynb    # Jupyter notebook for result exploration
└── requirements.txt      # Project dependencies

```

## 🚀 Usage

To run the full experimental suite, execute the main script. This will trigger the prequential evaluation on the defined datasets.

```bash
python run_experiment.py

```

*Note: By default, the script is configured to process 1,000,000 instances per dataset. You can modify the `INSTANCES` constant in `run_experiment.py` for quicker debugging.*

## 🧪 Experimental Setup

### Models Evaluated

The experiments compare the following variations of Hoeffding Tree Regressors (HTR):

1. **Baselines:**
* `HATR`: Hoeffding Adaptive Tree Regressor (Standard River implementation).
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

After running the experiments, logs are saved in the `logs/` directory. You can use the provided notebook `log_analysis.ipynb` or the scripts in `src/plot/` to generate:

* Performance over time plots.
* Critical Difference (CD) diagrams.
* Resource usage analysis (Memory/Time).

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{DIAS, C.G.,
  title={Adaptive Quantization Observers for Online Regression Trees},
  author={DIAS, Caio G.},
  journal={},
  year={2026}
}

```

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

# river-evaluate 📊

A lightweight framework for evaluating and comparing streaming learning models from the [River](https://riverml.xyz/) library, with a workflow inspired by Scikit-learn's `GridSearchCV`.

## Overview

This project provides a simple yet effective utility to benchmark different configurations of River's online machine learning models across multiple datasets. It was initially designed to explore the `HoeffdingTreeRegressor` but can be easily extended to other models.

The core of the evaluation uses River's `progressive_validation` methodology, which provides a robust measure of a model's performance on a data stream.

## Key Features

  - **Multi-Model Comparison**: Simultaneously evaluate multiple River models.
  - **Multi-Dataset Evaluation**: Test model performance across several datasets in a single run.
  - **Consolidated Results**: Summarize performance metrics (RMSE) in a clean `pandas` DataFrame.
  - **Simple Visualization**: Automatically generate bar plots to visually compare model performance.

-----

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/caiogimenes/river-evaluate.git
    cd river-evaluate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## ✨ Usage Guide

Below is a complete example of how to compare four different `HoeffdingTreeRegressor` configurations on the *Abalone* and *Wine Quality* datasets.

```python
import pandas as pd
from ucimlrepo import fetch_ucirepo
from river.tree import HoeffdingTreeRegressor
from src.evaluate import CompareModels

# 1. Load your datasets
# For this example, we fetch datasets from the UCI ML Repository
abalone = fetch_ucirepo(id=1)
wine_quality = fetch_ucirepo(id=186)

# 2. Prepare the data for the evaluator
# The evaluator expects a list of features (DataFrames) and targets (DataFrames/Series)
features_list = [abalone.data.features, wine_quality.data.features]
targets_list = [abalone.data.targets, wine_quality.data.targets]

# Create a map to label the datasets in the output
dataset_map = {
    0: 'abalone',
    1: 'wine_quality'
}

# 3. Define the River models you want to compare
models_list = [
    HoeffdingTreeRegressor(),
    HoeffdingTreeRegressor(leaf_prediction='mean'),
    HoeffdingTreeRegressor(grace_period=300),
    HoeffdingTreeRegressor(grace_period=100),
]

# Create a map to label the models in the output
models_map = {
    0: 'HTR_vanilla',
    1: 'HTR_mean',
    2: 'HTR_grace_300',
    3: 'HTR_grace_100',
}

# 4. Initialize and run the evaluator
evaluator = CompareModels(
    models=models_list,
    features=features_list,
    targets=targets_list,
    models_map=models_map,
    dataset_map=dataset_map
)

# 5. Get and display the results
# The summary_results() method returns a DataFrame with the final RMSE for each model-dataset pair
results_df = evaluator.summary_results()
print("Model Evaluation Results (RMSE):")
print(results_df)

# The plot_results() method generates and saves a comparison plot
evaluator.plot_results(filename='model_comparison_plot.png')
print("\nComparison plot saved to 'model_comparison_plot.png'")

```

### Example Output

Running the script above will produce a `pandas` DataFrame in your console and save a plot to your directory.

**Results DataFrame:**

```
Model Evaluation Results (RMSE):
               abalone  wine_quality
HTR_vanilla   2.231269      0.757018
HTR_mean      2.193134      0.741293
HTR_grace_300 2.193108      0.741293
HTR_grace_100 2.193231      0.741293
```

*(Note: Exact RMSE values may vary slightly between runs).*

**Comparison Plot:**

The generated plot provides a clear visual summary of the results, making it easy to identify the best-performing model for each dataset. To display it in your README, commit the image file to your repository and link to it.

*(**Tip**: After running the code, commit the `model_comparison_plot.png` file so the image is displayed correctly).*

-----

## Roadmap

Future plans for improving `river-evaluate` include:

  - [ ] **Support for Additional Metrics**: Add other key metrics like Mean Absolute Error (MAE) and R-squared ($R^2$).
  - [ ] **Broader Model Compatibility**: Officially support other regression and classification models from River.
  - [ ] **Unit Testing**: Implement a test suite to ensure code reliability and robustness.
  - [ ] **PyPI Publication**: Package the project for easy installation via `pip install river-evaluate`.

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
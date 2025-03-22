# Spaceship Titanic

This project is a solution to a competition from Kaggle and a production-ready ClearML-based solution based on it.

## The Data Science Part

Notebook with the best score: [notebook](https://drive.google.com/file/d/1BTe9fnHKcwIOmD4ghh7S7QAHY1pGXpwo/view?usp=sharing)\
Best score on Kaggle: **0.80827**

### Steps
- Data extraction and processing
- Feature engineering
- EDA
- Hyperparameters tuning
- Model training
- Model evaluating
- Blending model's results (only in this [notebook](https://drive.google.com/file/d/1BTe9fnHKcwIOmD4ghh7S7QAHY1pGXpwo/view?usp=sharing))

### Usage

1. Use Google Colab and configure folder structure:
```bash
ml_hits_1_module/ # Folder in main Google Drive directory
├── data/ # Folder for data from Kaggle
│ ├── train.csv # Train data
│ ├── test.csv # Test data
│ ├── sample_submission.csv # Example of a submission
├── notebooks/ # Folder for autosaved notebooks
├── submits/ # Folder for autosaved submissions
└── notebook.ipynb # Notebook with solution
```
2. Run all cells in notebook. \
   The submission file will be downloaded automatically and added to submits folder.

## MLOps part

It contains code for tuning, training and inference the model using the CLI and also tracking experiments using ClearML.

### Used tools
- Poetry
- ClearML
- Optuna
- Catboost
- Git

### Screenshots from local hosted ClearML

<table>
<tbody>
  <tr>
    <td>Hyperparameters tuning</td>
    <td>Training</td>
  </tr>
  <tr>
    <td><img src="https://github.com/clearml/clearml/blob/master/docs/experiment_manager.gif?raw=true" width="100%"></a></td>
    <td><img src="https://github.com/clearml/clearml/blob/master/docs/datasets.gif?raw=true" width="100%"></a></td>
  </tr>
  <tr>
    <td colspan="2" height="24px"></td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>Processed dataset</td>
  </tr>
  <tr>
    <td><img src="https://github.com/clearml/clearml/blob/master/docs/orchestration.gif?raw=true" width="100%"></a></td>
    <td><img src="https://github.com/clearml/clearml/blob/master/docs/pipelines.gif?raw=true" width="100%"></a></td>
  </tr>
</tbody>
</table>

## Author
- `Klyueva Victoria, 972302`

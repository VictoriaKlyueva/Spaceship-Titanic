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
    <td><img src="https://github.com/VictoriaKlyueva/Spaceship-Titanic/blob/refactoring/images/Tuning.png"></td>
    <td><img src="https://github.com/VictoriaKlyueva/Spaceship-Titanic/blob/refactoring/images/Training.png"></td>
  </tr>
  <tr>
    <td colspan="2" height="24px"></td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>Processed dataset</td>
  </tr>
  <tr>
    <td><img src="https://github.com/VictoriaKlyueva/Spaceship-Titanic/blob/refactoring/images/Inference.png"></td>
    <td><img src="https://github.com/VictoriaKlyueva/Spaceship-Titanic/blob/refactoring/images/Processed%20dataset.png"></td>
  </tr>
</tbody>
</table>

### How to run with CLI

1. Clone repository and move to project path
```bash
git clone https://github.com/VictoriaKlyueva/Spaceship-Titanic.git
cd spaceship_titanic
```

2. Install poetry
```bash
pip install poetry
```

3. Install project packages
```bash
poetry install --no-interaction --no-ansi
```

4. Configure your ClearML server running on `http://localhost/`

5. Follow the instructions and provide your credentials
```bash
clearml-init
```

6. Tune model
```bash
python model.py tune [--dataset=/path/to/train/dataset]
```

7. Train model
```bash
python model.py train [--dataset=/path/to/train/dataset]
```

8. Make predictions
```bash
python model.py predict [--dataset=/path/to/evaluation/dataset]
```

## Author
- `Klyueva Victoria, 972302`

import json
import optuna
from clearml import Task
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pandas as pd

from spaceship_titanic.utils.data_transformation import process_features
from spaceship_titanic.utils.logger import logger
from spaceship_titanic.config.constants import (
    SEED,
    CATEGORICAL_FEATURES
)


class ModelTuner:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def tune(self, n_trials=10):
        """
        Tune the model using Optuna and log results to ClearML.

        :param n_trials: Number of trials for hyperparameter optimization
        """
        task = Task.init(project_name='Spaceship Titanic', task_name='Hyperparameter tuning')

        # Use Optuna for hyperparameter optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial), n_trials=n_trials)

        # Log best parameters and accuracy
        task.get_logger().report_text(f"Best parameters: {study.best_params}")
        task.get_logger().report_scalar(
            title='Best accuracy',
            series='Best accuracy',
            value=study.best_value,
            iteration=0
        )

        # Save best hyperparameters as an artifact
        task.upload_artifact('best_hyperparameters', json.dumps(study.best_params))

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best accuracy: {study.best_value}")

        task.close()

    def _objective(self, trial):
        """
        Objective function for Optuna.

        :param trial: Optuna trial object
        :return: Model accuracy
        """
        task = Task.current_task()

        # Define hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': False,
            'cat_features': CATEGORICAL_FEATURES,
            'random_state': SEED
        }

        # Log hyperparameters to ClearML
        task.connect(params)

        # Load and preprocess data
        data = pd.read_csv(self.dataset_path)
        data = process_features(data)

        X = data.drop(['Transported'], axis=1)
        y = data['Transported']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=SEED)

        # Initialize and train the model
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        # Calculate accuracy
        accuracy = model.score(X_test, y_test)

        # Log accuracy to ClearML
        task.get_logger().report_scalar(
            title='Accuracy',
            series='Accuracy',
            value=accuracy,
            iteration=trial.number
        )

        return accuracy

import argparse
import json

from catboost import CatBoostClassifier
import pandas as pd
import joblib
import optuna
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split

from spaceship_titanic.utils.data_transformation import process_features
from spaceship_titanic.config import config_reader
from spaceship_titanic.utils.logger import logger
from spaceship_titanic.config.constants import (
    SEED,
    CATEGORICAL_FEATURES,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODEL_SAVE_PATH
)


# Class for working with model
class MyClassifierModel:
    def __init__(self, hyperparameters=None):
        self.target = ['Transported']
        self.hyperparameters = hyperparameters if hyperparameters else self._load_best_hyperparameters()
        self.model = CatBoostClassifier(**self.hyperparameters)

    @staticmethod
    def _load_best_hyperparameters():
        """
            Load best params from ClearML or use default values from poetry config if it absent
        """

        try:
            task = Task.get_task(project_name='Spaceship Titanic', task_name='Hyperparameter tuning')
            best_params = task.artifacts['best_hyperparameters'].get()

            best_params = json.loads(best_params)

            # Configure some other params
            best_params['cat_features'] = CATEGORICAL_FEATURES
            best_params['random_state'] = SEED

            logger.info("Hyperparameters loaded from ClearML")
            return best_params
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters from ClearML, params from poetry config will be used: {e}")
            return config_reader.load_hyperparameters_from_poetry()

    def train(self, dataset_path):
        """
            Train model on dataset from dataset_path

            :param dataset_path: path to train dataset
        """

        task = Task.init(project_name='Spaceship Titanic', task_name='Training model')

        # Load data
        logger.info(f"Loading dataset from {dataset_path}")
        try:
            data = pd.read_csv(dataset_path)
            data = process_features(data)
        except Exception as e:
            logger.error(f"An error occurred while loading the dataset: {e}")
            return

        X = data.drop(['Transported'], axis=1)
        y = data['Transported']

        # Train model
        logger.info("Training model...")
        self.model.fit(X, y)

        # Save model
        logger.info("Saving model")
        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Error occurred while saving: {e}")
            return

        # Upload model weights to ClearML
        logger.info("Uploading model weights to ClearML")
        try:
            task.upload_artifact(name='model_weights', artifact_object=MODEL_SAVE_PATH)
            logger.info("Model weights uploaded to ClearML")
        except Exception as e:
            logger.error(f"Error occurred while uploading model weights: {e}")

        task.close()

    def predict(self, dataset_path):
        """
            Test model on dataset from dataset_path

            :param dataset_path: path to test dataset
        """

        task = Task.init(project_name='Spaceship Titanic', task_name='Inference model')

        # Load data
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            data = pd.read_csv(dataset_path)
        except Exception as e:
            logger.error(f"An error occurred while loading the dataset: {e}")
            return

        result = data[['PassengerId']].copy()
        data = process_features(data)

        # Load model
        logger.info("Loading model")
        try:
            self.model = joblib.load('./data/model/model.pkl')
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            return

        # Save prediction
        logger.info("Making predictions...")
        try:
            predictions = self.model.predict(data)
        except Exception as e:
            logger.error(f"An error occurred while making prediction: {e}")
            return

        result['Transported'] = predictions

        result.to_csv('data/results.csv', index=False)

        # Upload prediction as artifact
        task.upload_artifact(name='submission', artifact_object='data/results.csv')

        logger.info("Predictions saved")

        task.close()


def tune_model(dataset_path):
    """
        Tuning model and log it to ClearML.

        :param dataset_path: path to train dataset
    """

    task = Task.init(project_name='Spaceship Titanic', task_name='Hyperparameter tuning')

    # Use optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset_path), n_trials=10)

    # Log best params
    task.get_logger().report_text(f"Best parameters: {study.best_params}")
    task.get_logger().report_scalar(title='Best accuracy', series='Best accuracy', value=study.best_value, iteration=0)

    # Save hyperparameters as artifact
    task.upload_artifact('best_hyperparameters', json.dumps(study.best_params))

    task.close()


def objective(trial, dataset_path):
    """
        Objective function for Optuna.

        :param trial: trial object Optuna
        :param dataset_path: path to train dataset
        :return: model accuracy
    """

    task = Task.current_task()

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

    # Logging model params ClearML
    task.connect(params)

    classifier_model = MyClassifierModel(params)

    # Load data
    data = pd.read_csv(dataset_path)
    data = process_features(data)

    X = data.drop(['Transported'], axis=1)
    y = data['Transported']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=SEED)

    # Train model
    classifier_model.model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test)
    )

    # Get score
    accuracy = classifier_model.model.score(X_test, y_test)

    # Logging metric to ClearML
    task.get_logger().report_scalar(title='Accuracy', series='Accuracy', value=accuracy, iteration=trial.number)

    return accuracy


if __name__ == '__main__':
    try:
        # Init argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train', 'predict', 'tune'])
        parser.add_argument('--dataset', required=False)

        args = parser.parse_args()

        # Set default paths
        if args.command == 'train' and args.dataset is None:
            args.dataset = TRAIN_DATA_PATH
        elif args.command == 'predict' and args.dataset is None:
            args.dataset = TEST_DATA_PATH
        elif args.command == 'tune' and args.dataset is None:
            args.dataset = TRAIN_DATA_PATH
    except Exception as e:
        logger.error(f"An error occurred while parsing command: {e}")
        raise

    # Start model
    model = MyClassifierModel()
    if args.command == 'train':
        model.train(args.dataset)
    elif args.command == 'predict':
        model.predict(args.dataset)
    elif args.command == 'tune':
        tune_model(args.dataset)

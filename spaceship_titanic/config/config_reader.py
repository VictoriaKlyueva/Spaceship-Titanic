import toml
from pathlib import Path


def load_hyperparameters_from_poetry():
    """
        Получение гиперпараметров модели из pyproject.toml
    """
    # Путь к pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

    # Чтение файла
    with open(pyproject_path, "r") as f:
        config = toml.load(f)

    # Извлечение гиперпараметров
    hyperparameters = config.get("tool", {}).get("model", {}).get("hyperparameters", {})
    return hyperparameters


if __name__ == "__main__":
    hyperparams = load_hyperparameters_from_poetry()
    print(hyperparams)
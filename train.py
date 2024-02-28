from config import UUID, PROJECT_ROOT_PATH
from src.model import Model


def main():
    model_kwargs = {'n_estimators': 100}
    model = Model(PROJECT_ROOT_PATH, UUID, **model_kwargs)
    model.fit()

    model.dump()

if __name__ == "__main__":
    main()
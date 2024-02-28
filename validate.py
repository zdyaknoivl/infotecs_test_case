from config import UUID, PROJECT_ROOT_PATH
from src.model import Model


def main():
    model = Model(PROJECT_ROOT_PATH, UUID)
    model.write_val_results()
    
if __name__ == "__main__":
    main()
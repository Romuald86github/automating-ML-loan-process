import os

def create_project_structure():
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "src/data",
        "src/features",
        "src/models",
        "src/app"
    ]
    
    files = [
        "src/data/data_loader.py",
        "src/data/data_cleaning.py",
        "src/features/feature_engineering.py",
        "src/features/preprocessing.py",
        "src/models/train_model.py",
        "src/app/app.py",
        "run_project.py",
        "requirements.txt",
        "README.md"
    ]
    
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    
    for file in files:
        with open(file, 'w') as f:
            pass

if __name__ == "__main__":
    create_project_structure()

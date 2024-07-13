import os
import logging

from pathlib import Path

# logging string

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'talentOptimize'

list_of_files = [
    ".github/workflows/.gitkeep",
    "requirements.txt",
    "app.py",
    f"recruitment_system/src/{project_name}/__init__.py",
    f"recruitment_system/src/{project_name}/components/__init__.py",
    f"recruitment_system/src/{project_name}/utils/__init__.py",
    f"recruitment_system/src/{project_name}/config/__init__.py",
    f"recruitment_system/src/{project_name}/config/configuration.py",
    f"recruitment_system/src/{project_name}/pipeline/__init__.py",
    f"recruitment_system/src/{project_name}/entity/__init__.py",
    f"recruitment_system/src/{project_name}/constants/__init__.py",
    "recruitment_system/config/config.yaml",
    "recruitment_system/research/trials.ipynb",
    "recruitment_system/params.yaml",
    "recruitment_system/requirements.txt",
    "recruitment_system/main.py",
    "setup.py",
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")
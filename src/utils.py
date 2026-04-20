import os

DATA_PATH = "./data"

def get_subjects_structure():
    """Scans the data folder to populate the Sidebar Dropdowns dynamically."""
    structure = {}
    if os.path.exists(DATA_PATH):
        for sem in sorted(os.listdir(DATA_PATH)):
            sem_path = os.path.join(DATA_PATH, sem)
            if os.path.isdir(sem_path) and not sem.startswith("."):
                subjects = []
                for subj in sorted(os.listdir(sem_path)):
                    if not subj.startswith("."):
                        subjects.append(subj)
                structure[sem] = subjects
    return structure

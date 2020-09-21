from pathlib import Path, PosixPath
import json

def mkdirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def make_dataset_of_files(root, extensions=['.npy']):
    """The root of dataset contains files of the given extension."""
    root = Path(root)
    assert root.is_dir(), f"{root} is not a valid directory"
    paths = [root / file for file in root.iterdir() if has_extension(file, extensions)]
    return sorted(paths)
    
def has_extension(file, extensions):
    if isinstance(file, str):
        return any(file.endswith(ext) for ext in extensions)
    elif isinstance(file, PosixPath):
        return any(ext in extensions for ext in file.suffixes)

def make_dataset_of_directories(root, extensions=['.npy']):
    """The root of dataset contains folders for each data point. Each data point folder has to have
    (at least) one file of the specified extension. The dataset has to define which file it takes from
    such folder. Useful when using a dataset that stores, for example, an image and a mask together
    in their own folder.
    """
    root = Path(root)
    assert root.is_dir(), f"{root} is not a valid directory"
    paths = [root / folder for folder in root.iterdir() if (root / folder).is_dir()]
    paths = [folder for folder in paths if has_files_with_extension(folder, extensions)]
    return sorted(paths)

def has_files_with_extension(folder, extensions):
    for ext in extensions:
        if not ext.startswith("."):
            ext = "." + ext
        files_in_folder = list(folder.glob(f"*{ext}"))
        if files_in_folder:
            return True
    return False

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
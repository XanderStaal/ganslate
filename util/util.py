import os
import json


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def has_extension(file, extensions):
    return any(file.endswith(ext) for ext in extensions)

def make_dataset(root, extensions=['.npy']):
    assert os.path.isdir(root), '%s is not a valid directory' % dir
    paths = [os.path.join(root, file) for file in os.listdir(root) if has_extension(file, extensions)]
    return paths

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
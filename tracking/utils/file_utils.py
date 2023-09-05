import numpy as np
import cv2
import os


def make_dirs(folder_path, **kargs):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path + "/", **kargs)

def remove_dir(folder_path):
    os.system("rm -rf " + folder_path)

def remove_file(file_path):
    os.system("rm -f " + file_path)

def dir_name(file_path):
    return os.path.dirname(file_path)

def file_name(file_path):
    return os.path.basename(file_path)

def is_file(file_path):
    return os.path.isfile(file_path)

def is_dir(dir_path):
    return os.path.isdir(dir_path)

def get_files(path, extensions):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extensions):
                matches.append(os.path.join(root, filename))
    return matches

def get_sub_folders(folderPath):
    subfolders = [f.path for f in os.scandir(folderPath) if f.is_dir()]
    return subfolders
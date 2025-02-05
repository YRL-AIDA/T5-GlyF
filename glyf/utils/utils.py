import json
import pickle
import argparse

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def save_pkl(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def parse_args():
    parser = argparse.ArgumentParser(description="Process the data: analyze and corrupt")
    parser.add_argument('--config_path', type=str, required=True, help='Input path to the configuration file')
    return parser.parse_args()
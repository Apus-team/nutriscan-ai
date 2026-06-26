import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_labels():
    path = os.path.join(BASE_DIR, "labels.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_calories():
    path = os.path.join(BASE_DIR, "calories.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label(index):
    labels = load_labels()
    return labels.get(str(index), "unknown")


def get_calories(class_name):
    calories = load_calories()
    return calories.get(class_name, None)

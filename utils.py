import json
import random

import numpy as np
import pandas as pd
import torch
import string

def unique(lst):
    # insert the list to the set
    list_set = set(lst)
    # convert the set to the list
    unique_list = list(list_set)
    random.shuffle(unique_list)
    return unique_list


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def column(matrix, i):
    return [row[i] for row in matrix]


def read_json(name, batch_size):
    with open(name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    current_batch_idx = len(data['references'])//batch_size
    # if 'fewshot' in data:
    #     fewshot = data['fewshot']
    # else:
    #     fewshot = None
    # try:
    #     del data['fewshot']
    # except:
    #     pass
    # df = pd.DataFrame(data)
    return data, current_batch_idx

def save_to_json(data, name):
    jsonString = json.dumps(data, indent=4, ensure_ascii=False)
    jsonFile = open(name, "w", encoding="utf-8")
    jsonFile.write(jsonString)
    jsonFile.close()


def save_to_csv(data, name):
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(text)))

import pickle
import numpy as np

from numpy import ndarray
from typing import List

def cosine(vec1: ndarray, vec2:ndarray):
    return vec1.dot(vec2)/np.linalg.norm(vec1)*np.linalg.norm(vec2)

def compute(item_emb_1 : List[float], item_emb_2 : List[float]) -> float:
    return cosine(np.array(item_emb_1),  np.array(item_emb_2))

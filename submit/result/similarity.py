from typing import List
def compute(item_emb_1 : List[float], item_emb_2 : List[float]) -> float:
    s = sum([a*b for a,b in zip(item_emb_1, item_emb_2)])
    return s
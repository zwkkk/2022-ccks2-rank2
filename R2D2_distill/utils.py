import json
import pandas as pd

from pandas import DataFrame

def read_data(filename: str) -> DataFrame:
    with open(filename) as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines]
    data = pd.DataFrame(data)
    return data

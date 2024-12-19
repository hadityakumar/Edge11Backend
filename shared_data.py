import pandas as pd
from functools import lru_cache

# Lazy loading with caching
@lru_cache(maxsize=1)
def get_data():
    print("Loading dataset...")
    # Load the Parquet file once
    data = pd.read_parquet("predict_dataset.parquet")
    return data

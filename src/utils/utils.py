import pandas as pd


def memory_usage(obj):
    if isinstance(obj, pd.DataFrame):
        usage_b = obj.memory_usage(deep=True).sum()
    else:
        usage_b = obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024**2
    return f"{usage_mb:03.2f} MB"

import pandas as pd


def memory_usage(obj):
    if isinstance(obj, pd.DataFrame):
        usage_b = obj.memory_usage(deep=True).sum()
    else:
        usage_b = obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024**2
    return f"{usage_mb:03.2f} MB"


def get_optimised_data(data: pd.DataFrame) -> pd.DataFrame:

    dtype_mapping = {"float": "float", "int": "unsigned"}

    optimised_data = data.copy()

    for dtype in ["float", "int", "str"]:
        if dtype == "float" or dtype == "int":
            data_dtype = data.select_dtypes(include=dtype)
            data_converted_dtype = data_dtype.apply(
                pd.to_numeric, downcast=dtype_mapping[dtype]
            )
        else:
            obj_columns = data.select_dtypes(include=["str"]).columns.to_list()
            for col in obj_columns:
                optimised_data[col] = data[col].astype("category")

        optimised_data[data_converted_dtype.columns] = data_converted_dtype

    return optimised_data


def set_run_id(model_name):
    from datetime import UTC, datetime

    run_id = f"{datetime.now(tz=UTC).strftime(format='%Y%m%d-%H%M%S')}_{model_name}"
    return run_id

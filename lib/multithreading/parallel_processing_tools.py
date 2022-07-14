from concurrent.futures import ProcessPoolExecutor
import pandas as pd


def execute_function_in_parallel(func, number_of_processors, args_list):
    pool = ProcessPoolExecutor(number_of_processors)

    futures = [
        pool.submit(func, **args) if type(args) == dict else pool.submit(func, *args)
        for args in args_list
    ]

    results = [
        f.result() for f in futures
    ]

    print("All processes finished!")

    return results


def execute_function_sequentially(func, args_list):
    print("Sequential computation.")

    return [
        func(**args) if type(args) == dict else func(*args) for args in args_list
    ]


def apply_function_to_list_of_args_and_concat_resulting_dfs(func, args_list, number_of_processors, concat_axis=None):
    if number_of_processors is None or number_of_processors > 1:
        res = execute_function_in_parallel(func, number_of_processors, args_list)
    else:
        res = execute_function_sequentially(func, args_list)

    if concat_axis is not None:
        print('Concatenating results of parallel processes.')
        return pd.concat(res, axis=concat_axis)

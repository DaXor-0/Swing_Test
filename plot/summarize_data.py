import os
import sys
import re
import argparse
import tarfile
import numpy as np
import pandas as pd

DTYPE_TO_BYTES = {
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8,
    'float': 4,
    'double': 8,
    'char': 1,
    'int': 4
}


def process_benchmark_file(file_path, warmup_ratio=0.2):
    """
    Read a benchmark CSV file, discard the warmup iterations,
    extract the 'highest' column, and compute summary statistics.
    """
    df = pd.read_csv(file_path)
    warmup_count = int(len(df) * warmup_ratio)
    df = df.iloc[warmup_count:]

    highest = df['highest'].values
    n_iter = len(highest)

    mean = np.mean(highest)
    median = np.median(highest)
    std = np.std(highest, ddof=1)
    var = np.var(highest, ddof=1)
    min_val = np.min(highest)
    max_val = np.max(highest)

    percentile_10 = np.percentile(highest, 10)
    percentile_25 = np.percentile(highest, 25)
    percentile_75 = np.percentile(highest, 75)
    percentile_90 = np.percentile(highest, 90)
    iqr = percentile_75 - percentile_25

    # Standard Error and Confidence Interval (95% CI)
    if n_iter > 1:
        standard_error = std / np.sqrt(n_iter)
        ci_lower = mean - 1.96 * standard_error
        ci_upper = mean + 1.96 * standard_error
    else:
        standard_error = None
        ci_lower, ci_upper = None, None

    # Outlier detection using the IQR method
    outliers = highest[(highest < (percentile_25 - 1.5 * iqr))
                       | (highest > (percentile_75 + 1.5 * iqr))]
    num_outliers = len(outliers)

    stats = {
        'mean': mean,
        'median': median,
        'std': std,
        'var': var,
        'min': min_val,
        'max': max_val,
        'n_iter': n_iter,
        'percentile_10': percentile_10,
        'percentile_25': percentile_25,
        'percentile_75': percentile_75,
        'percentile_90': percentile_90,
        'iqr': iqr,
        'standard_error': standard_error,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        # 'outliers': list(outliers),
        'num_outliers': num_outliers
    }
    return stats


def parse_filename(filename):
    """
    Parse the filename to extract the array dimension, algorithm name, and data type.
    Expected format: <array_dim>_<algo_name>_<dtype>.csv
    Example: 512_swing_static_over_int32.csv
    """
    pattern = r"(?P<array_dim>\d+)_(?P<algo_name>.+)_(?P<dtype>[^_]+)\.csv"
    match = re.match(pattern, filename)
    if match:
        params = match.groupdict()
        try:
            array_dim = int(params['array_dim'])
            dtype = params['dtype']
            dtype_size = DTYPE_TO_BYTES.get(dtype, 4)  # Default to 4 bytes if unknown
            buffer_size = array_dim * dtype_size
        except (ValueError, KeyError) as e:
            print(f"Error processing filename {filename}: {str(e)}", file=sys.stderr)
            buffer_size = None
        
        return {
            'array_dim': array_dim,
            'algo_name': params['algo_name'],
            'dtype': dtype,
            'buffer_size': buffer_size
        }
    else:
        return {'array_dim': None, 'algo_name': None, 'dtype': None, 'buffer_size': None}


def aggregate_results(results_dir: os.PathLike, metadata: pd.DataFrame, target_timestamp: str, system_name: str) -> pd.DataFrame:
    """
    Walk thorught restults_dir subdirectory and process each .csv file to aggregate results.
    Beware of metadata file.
    """
    all_results = []

    for test_id, meta_row in metadata.iterrows():
        test_dir = os.path.join(results_dir, str(test_id))
        if not os.path.isdir(test_dir):
            print(f"Test directory {test_dir} not found.", file=sys.stderr)
            continue

        for file in os.listdir(test_dir):
            if file.endswith(".csv"):
                filepath = os.path.join(test_dir, file)
                stats = process_benchmark_file(filepath)
                file_params = parse_filename(file)

                result = {
                    'collective_type': meta_row['collective_type'],
                    'array_dim': file_params.get('array_dim'),
                    'buffer_size': file_params.get('buffer_size'),
                    'algo_name': file_params.get('algo_name'),
                    'datatype': file_params.get('dtype'),
                    'mean': stats['mean'],
                    'median': stats['median'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'nnodes': meta_row['nnodes'],
                    'n_iter': stats['n_iter'],
                    'percentile_10': stats['percentile_10'],
                    'percentile_25': stats['percentile_25'],
                    'percentile_75': stats['percentile_75'],
                    'percentile_90': stats['percentile_90'],
                    'iqr': stats['iqr'],
                    'standard_error': stats['standard_error'],
                    'ci_lower': stats['ci_lower'],
                    'ci_upper': stats['ci_upper'],
                    # 'outliers': stats['outliers'],
                    'num_outliers': stats['num_outliers'],
                    'system': system_name,
                    'timestamp': target_timestamp,
                    'test_id': str(test_id),
                    'MPI_Op': meta_row['MPI_Op'],
                    'CUDA': meta_row['CUDA'],
                    'notes': meta_row['notes'],
                    'mpi_lib': meta_row['mpi_lib'],
                    'mpi_lib_version': meta_row['mpi_lib_version'],
                    'libswing_version': meta_row['libswing_version']
                }
                all_results.append(result)
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--result-dir", required=True, help="Full path to the test results")
    args = parser.parse_args()

    # Ensure the path follows the expected format: results/<system>/<timestamp>
    this_results_dir = os.path.normpath(args.result_dir)
    parts = this_results_dir.split(os.sep)
    if len(parts) < 3 or parts[-3] != "results":
        print(f"Invalid result directory structure: {this_results_dir}", file=sys.stderr)
        sys.exit(1)

    location, timestamp = parts[-2], parts[-1]
    tar_path = this_results_dir + ".tar.gz"

    # If result dir exists only as a tar.gz and not as a directory, extract it
    if not os.path.isdir(this_results_dir):
        if os.path.isfile(tar_path):
            print(f"Extracting {tar_path} to {os.path.dirname(this_results_dir)} for processing...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(this_results_dir))
        else:
            print(f"Directory {this_results_dir} does not exist and no tar.gz file found.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Directory {this_results_dir} exists. Using it for processing...")

    metadata_path = f"results/{location}_metadata.csv"
    if not os.path.isfile(metadata_path):
        print(f"Metadata file {metadata_path} not found.", file=sys.stderr)
        sys.exit(1)
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.set_index(["timestamp", "test_id"], inplace=True)
    test_metadata = metadata_df.loc[timestamp]

    # Aggregate all results by processing each benchmark file
    aggregated_df = aggregate_results(
        this_results_dir, test_metadata, timestamp, location)

    # Display the first few rows of the aggregated results
    print("Aggregated Data:")
    print(aggregated_df.head())

    # Optionally, save the aggregated summary to a CSV file
    aggregated_df.to_csv(
        f"{this_results_dir}/aggregated_results_summary.csv", index=False)


if __name__ == '__main__':
    main()

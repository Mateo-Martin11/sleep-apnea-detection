"""Generate submission CSV files."""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_submission(
    predictions: np.ndarray,
    sample_ids: np.ndarray,
    output_path: str = "submissions/submission.csv",
) -> pd.DataFrame:
    """Save predictions as a challenge-compliant CSV.

    Args:
        predictions: (N, 90) binary int array
        sample_ids:  (N,) int array of sample IDs
        output_path: where to save the file

    Returns:
        The resulting DataFrame.
    """
    if predictions.dtype != int:
        predictions = (predictions >= 0.5).astype(int)

    columns = ["ID"] + [f"y_{i}" for i in range(90)]
    data = np.column_stack([sample_ids, predictions])
    df = pd.DataFrame(data, columns=columns)
    df["ID"] = df["ID"].astype(int)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}  ({len(df)} rows)")
    return df

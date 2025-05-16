from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument("hsc_path", type=Path)
    hsc_path = parser.parse_args().hsc_path
    results = []
    for results_path in hsc_path.glob("output/*/*/results.csv"):
        df = pd.read_csv(results_path)
        results.append({
            "solution": results_path.parent.parent.name,
            "level": results_path.parent.name.replace("Test_", ""),
            "split": "test" if "Test" in str(results_path) else "train",
            "mean-cer": float(df["CER"].mean()),
        })
    print(f"Found {len(results)} results")
    df = pd.DataFrame(results).sort_values(by=["solution", "level", "split"], ignore_index=True)
    df.to_csv(out := hsc_path / "joined_results.csv")
    print(f"Wrote output results to {out.resolve()}")


if __name__ == "__main__":
    main()

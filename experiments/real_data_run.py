"""Run the pipeline on the real smartphone driving sessions.

Windows the labelled day sessions, extracts features, and evaluates the model against
the fixed-threshold baseline and the majority-class floor under leave-one-group-out,
where the group is the recording day. Two window lengths are compared: the two-second
window adopted after Banos et al. (2014) and the thirty-second window the proposal
originally specified.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import compare_models, default_model, leave_one_group_out  # noqa: E402
from src.features import features_table  # noqa: E402
from src.load_mendeley import ACC_COLUMNS, SIGNAL_COLUMNS, load_sessions  # noqa: E402
from src.windowing import WindowSpec, estimate_rate_hz, make_windows  # noqa: E402

RATE_HZ = 100.0


def main(root: str = "data/mendeley") -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    print("=== Sessions ===")
    raw = load_sessions(root)
    print(f"\nTotal samples: {len(raw):,}")
    print(f"Sessions: {raw['group'].nunique()}  Days: {sorted(raw['day'].unique())}")
    print(f"Estimated rate: {estimate_rate_hz(raw['t']):.1f} Hz")
    print(raw.groupby(["label"])["group"].nunique().to_string())

    # Two of the twelve sessions (Day-2R and Day-6R) ship no gyroscope file, and the
    # loader zero-fills those channels. Both are rash sessions, so "gyroscope is exactly
    # zero" is a perfect giveaway for the positive class in any fold that contains them.
    # The accelerometer-only configuration removes that shortcut; both are reported.
    configs = [
        ("accelerometer only", ACC_COLUMNS, (2.0, 30.0)),
        ("accelerometer + gyroscope (leaky)", SIGNAL_COLUMNS, (2.0,)),
    ]

    for config_name, columns, lengths in configs:
      print(f"\n########## {config_name} ##########")
      for length_s in lengths:
        spec = WindowSpec(length_s=length_s, overlap=0.5)
        windows = make_windows(raw, columns, "t", "label", "group", spec)
        table = features_table(windows, columns, RATE_HZ)
        tag = "acc" if columns is ACC_COLUMNS else "accgyro"
        if table.empty:
            print(f"\n{length_s:g} s: no windows survived the fill and purity checks")
            continue

        # Group by day so that both sessions of a day are held out together: the rash and
        # safe runs of one day share a driver, a vehicle and a phone placement, and
        # leaving only one of them out would let the model see that day's conditions.
        table["group"] = table["group"].str.extract(r"Day-(\d+)")[0].astype(int)

        print(f"\n=== {length_s:g} s windows ===")
        print(f"{len(table)} windows, {len(table.columns) - 3} features")
        print(table.groupby(["group", "label"]).size().unstack(fill_value=0).to_string())

        comparison = compare_models(table)
        print("\nLeave-one-day-out:")
        print(comparison.round(3).to_string(index=False))
        comparison.to_csv(out_dir / f"real_comparison_{tag}_{int(length_s)}s.csv", index=False)

        if length_s == 2.0:
            detail = leave_one_group_out(table, default_model())
            print("\nPer-day folds (gradient-boosted trees):")
            print(detail["folds"].round(3).to_string(index=False))
            detail["folds"].to_csv(out_dir / f"real_folds_{tag}_2s.csv", index=False)
            print("\nPooled confusion matrix (rows = true):")
            print(detail["confusion"].to_string())
            detail["confusion"].to_csv(out_dir / f"real_confusion_{tag}_2s.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/mendeley")

"""Measure whether the fallback dataset can carry the project's modelling claim.

The proposal named a synthetic trip-level eco-driving dataset as the fallback if access
to real recordings was delayed. This script tests what that fallback can actually
support, before any work is built on it.

The test is simple: if a plain linear model predicts the target as well as a flexible
nonlinear one, there is no structure to learn beyond the formula the data was generated
from, and any "model" trained on it is recovering generator coefficients.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "eco_score"


def main(path: str) -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}  Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # Structural check first: what the file does and does not contain.
    has_time = any(c.lower() in {"t", "time", "timestamp"} for c in df.columns)
    def looks_like_identifier(column: str) -> bool:
        name = column.lower()
        if any(k in name for k in ("driver", "trip", "vehicle", "subject")):
            return True
        # Match an "id" token, not the letters i-d inside a word such as "idling_time".
        return name == "id" or name.endswith("_id") or name.startswith("id_")

    has_group = any(looks_like_identifier(c) for c in df.columns)
    print(f"\nPer-sample timestamp column present: {has_time}")
    print(f"Driver or trip identifier present:    {has_group}")
    print("Consequence: no per-sample signal means no windowing, and no identifier "
          "means leave-one-driver-out cannot be run on this file.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rows = []
    for name, model in [
        ("Linear regression", LinearRegression()),
        ("Gradient boosting", GradientBoostingRegressor(random_state=0)),
    ]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "R2": r2_score(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
            }
        )
    table = pd.DataFrame(rows)
    print("\nHeld-out prediction of the composite target:")
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / "eco_dataset_check.csv", index=False)

    linear = LinearRegression().fit(X, y)
    coefs = pd.Series(linear.coef_, index=X.columns).round(4)
    residual = y - linear.predict(X)
    print("\nRecovered linear coefficients:")
    print(coefs.to_string())
    print(f"Intercept: {linear.intercept_:.3f}")
    print(f"Residual std: {residual.std():.3f}   Target std: {y.std():.3f}")

    gap = table.loc[table.Model == 'Gradient boosting', 'R2'].iloc[0] - \
        table.loc[table.Model == 'Linear regression', 'R2'].iloc[0]
    print(f"\nNonlinear minus linear R2: {gap:+.4f}")
    print("A flexible model that cannot beat a straight line indicates the target is a "
          "weighted sum of the features plus noise. The residual spread is the injected "
          "noise, and it is the irreducible error.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/eco_driving_score.csv")

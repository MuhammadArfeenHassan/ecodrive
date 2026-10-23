# On-Device Driving Behavior Classification

CSCI 49000-AIT semester project. Muhammad Arfeen Hassan (hassa101@purdue.edu).

Classifies windows of smartphone inertial signal as rash or safe
driving, with inference intended to run on the device in the vehicle rather than in the
cloud.

## State of the code, as of the Week 9 progress report

**Runs end to end:** loading the real sessions, windowing, feature extraction, the
leave-one-group-out evaluation harness, the fixed-threshold baseline and the
majority-class floor.

**Not yet done:** trip-level aggregation (the "wisdom" layer), the TensorFlow Lite
export, and any model-size or latency measurement.

## Layout

```
src/load_mendeley.py   loader for the smartphone driving sessions, with clock repair
src/windowing.py       segment per-sample recordings into overlapping windows
src/features.py        time-domain and spectral features per window
src/evaluate.py        leave-one-group-out harness, threshold baseline, comparison
experiments/real_data_run.py          the real result (Table 1 of the report)
experiments/eco_dataset_check.py      measures what the fallback dataset can support
experiments/simulated_smoke_test.py   pipeline check on generated signal
results/               CSV outputs
data/                  input files (not committed)
```

## Running

```bash
pip install -r requirements.txt
# extract the Day-*.rar folders into data/mendeley/ first
python3 experiments/real_data_run.py data/mendeley
```

## Results

Leave-one-day-out over 7 days, 2 s windows, accelerometer only, 9,530 windows
(2,381 rash / 7,149 safe) from 972,755 samples:

| Model | Pooled accuracy | Pooled macro-F1 | Mean fold F1 | Fold std |
|---|---|---|---|---|
| Gradient-boosted trees | 0.853 | **0.787** | 0.640 | 0.263 |
| Fixed-threshold baseline | 0.734 | 0.283 | 0.336 | 0.115 |
| Majority class | 0.750 | 0.429 | 0.580 | 0.291 |

Three caveats that belong next to those numbers:

- **Days 3, 5 and 7 contain almost no rash driving** (3, 0 and 0 rash windows), so those
  folds compute macro-F1 over a single class. That is why mean fold F1 sits well below
  the pooled figure.
- **Day-2R and Day-6R have no gyroscope file** and are zero-filled by the loader. Both
  are rash sessions, so gyroscope features leak the label: including them lifts pooled
  macro-F1 from 0.787 to 0.808. The headline configuration is accelerometer-only.
- **The millisecond column is unusable as written.** One session has 111,212 rows under
  7,663 distinct stamps, with gaps up to 260 s, because the logger stalls and flushes.
  The loader rebuilds the clock from acquisition order and cuts at every stall.

The dataset has no drowsy sessions, so this is a two-class problem (rash vs safe) rather
than the three-class one the proposal specified.

## The fallback dataset, and why it is not used for modelling

On the synthetic trip-level eco-driving file, linear regression predicts the target at
R² 0.9339 while gradient boosting reaches 0.9295. A flexible model that cannot beat a
straight line means the target is a weighted sum plus noise. It also has no per-sample
timestamp and no driver identifier, so it cannot be windowed or grouped.

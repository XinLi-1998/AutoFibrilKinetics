import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

BASE = Path(r'c:\Users\17865\Desktop\GCG Mix')
DATA_DIR = BASE / '3.0-3'
TRP_FILE = DATA_DIR / 'Trp.xlsx'
THT_FILE = DATA_DIR / 'ThT.xlsx'
OUT_XLSX = DATA_DIR / 'lag_time_results_initial.xlsx'
OUT_CSV = DATA_DIR / 'lag_time_results_initial.csv'
SAMPLES = [f'G{i}' for i in range(22)]
IGNORE_SUB = {'AVR', 'SD', 'SD.1'}


def load_book(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=[0, 1])
    cols = []
    for a, b in df.columns:
        a = 'G20' if a == 'GCG20' else a
        cols.append((a, b))
    df.columns = pd.MultiIndex.from_tuples(cols)
    return df


def get_time_minutes(df: pd.DataFrame) -> np.ndarray:
    t = pd.to_timedelta(df.iloc[:, 0].astype(str))
    return t.dt.total_seconds().to_numpy() / 60.0


def sample_replicates(df: pd.DataFrame, sample: str):
    cols = [b for a, b in df.columns if a == sample and b not in IGNORE_SUB]
    return cols[:3]


def compute_lag(t: np.ndarray, y: np.ndarray, mode: str):
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t = t[ok]
    y = y[ok]
    if len(y) < 9:
        raise ValueError('Not enough points')

    window = min(9, len(y) if len(y) % 2 == 1 else len(y) - 1)
    ys = savgol_filter(y, window_length=window, polyorder=3)
    baseline = float(np.median(ys[:4]))
    if mode == 'trp':
        signal = baseline - ys
        raw_signal = baseline - y
    else:
        signal = ys - baseline
        raw_signal = y - baseline

    amp = float(np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5))
    early_sd = float(np.nanstd(raw_signal[:6], ddof=1)) if len(raw_signal) >= 6 else float(np.nanstd(raw_signal, ddof=1))
    snr = amp / max(early_sd, 1e-6)

    fine_t = np.linspace(float(t.min()), float(t.max()), int((t.max() - t.min()) * 10) + 1)
    fine_signal = PchipInterpolator(t, signal)(fine_t)
    slope = np.gradient(fine_signal, fine_t)
    lo = 0.05 * amp
    hi = 0.70 * amp if mode == 'trp' else 0.60 * amp
    mask = (fine_signal >= lo) & (fine_signal <= hi)
    if not np.any(mask):
        mask = (fine_signal >= 0) & (fine_signal <= 0.85 * amp)
    masked_slope = np.where(mask, slope, -np.inf)
    idx = int(np.argmax(masked_slope))
    inflect_t = float(fine_t[idx])
    inflect_signal = float(fine_signal[idx])
    inflect_slope = float(slope[idx])
    lag = inflect_t - inflect_signal / inflect_slope if inflect_slope > 0 else np.nan
    tail = float(np.median(signal[-5:]))
    tail_progress = tail / amp if amp > 0 else np.nan
    valid = bool(np.isfinite(lag) and (-30 <= lag <= t.max() + 30) and amp > 0 and inflect_slope > 0)

    thresholds = {}
    for frac in (0.1, 0.2, 0.5):
        idxs = np.where(fine_signal >= frac * amp)[0]
        thresholds[f't{int(frac*100)}'] = float(fine_t[idxs[0]]) if len(idxs) else np.nan

    return {
        'lag_min': float(lag),
        'snr': snr,
        'amplitude': amp,
        'tail_progress': tail_progress,
        'valid': valid,
        't10': thresholds['t10'],
        't20': thresholds['t20'],
        't50': thresholds['t50'],
    }


def combine_lag(trp_res, tht_res):
    trp = trp_res['lag_min']
    tht = tht_res['lag_min']
    use_tht = False
    reason = 'Trp primary'
    if tht_res['valid'] and tht_res['snr'] >= 4 and tht_res['tail_progress'] >= 0.35 and np.isfinite(tht):
        delta = abs(tht - trp)
        if delta <= 20:
            combined = 0.75 * trp + 0.25 * tht
            use_tht = True
            reason = 'Trp-dominant weighted blend'
        elif delta <= 40:
            combined = 0.85 * trp + 0.15 * tht
            use_tht = True
            reason = 'Trp-dominant weak ThT correction'
        else:
            combined = trp
            reason = 'ThT inconsistent, ignored'
    else:
        combined = trp
        reason = 'ThT low-confidence, ignored'
    return float(combined), use_tht, reason


def detect_outliers(df):
    flagged = []
    for sample, g in df.groupby('sample', sort=False):
        vals = g['combined_lag_min'].to_numpy(dtype=float)
        med = float(np.median(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        for _, row in g.iterrows():
            delta = abs(float(row['combined_lag_min']) - med)
            if delta >= 20 or (sd > 0 and delta >= max(1.2 * sd, 12)):
                flagged.append({
                    'sample': sample,
                    'replicate': int(row['replicate']),
                    'combined_lag_min': float(row['combined_lag_min']),
                    'sample_median': round(med, 2),
                    'sample_sd': round(sd, 2),
                    'delta_from_median': round(delta, 2),
                })
    return pd.DataFrame(flagged)


def main():
    trp_df = load_book(TRP_FILE)
    tht_df = load_book(THT_FILE)
    t = get_time_minutes(trp_df)
    print('TRP', trp_df.shape, 'THT', tht_df.shape, 'TIMEPOINTS', len(t), 'TIME_END_MIN', round(float(t.max()), 2))

    rows = []
    for sample in SAMPLES:
        trp_reps = sample_replicates(trp_df, sample)
        tht_reps = sample_replicates(tht_df, sample)
        print(sample, trp_reps, tht_reps)
        for idx, (trp_col, tht_col) in enumerate(zip(trp_reps, tht_reps), start=1):
            trp_res = compute_lag(t, trp_df[(sample, trp_col)].to_numpy(), 'trp')
            tht_res = compute_lag(t, tht_df[(sample, tht_col)].to_numpy(), 'tht')
            combined_lag, used_tht, reason = combine_lag(trp_res, tht_res)
            rows.append({
                'sample': sample,
                'replicate': idx,
                'trp_col': trp_col,
                'tht_col': tht_col,
                'trp_lag_min': round(trp_res['lag_min'], 2),
                'tht_lag_min': round(tht_res['lag_min'], 2),
                'combined_lag_min': round(combined_lag, 2),
                'tht_used': used_tht,
                'decision': reason,
                'trp_t10': round(trp_res['t10'], 2),
                'trp_t20': round(trp_res['t20'], 2),
                'trp_t50': round(trp_res['t50'], 2),
                'tht_t10': round(tht_res['t10'], 2),
                'tht_t20': round(tht_res['t20'], 2),
                'tht_t50': round(tht_res['t50'], 2),
                'trp_snr': round(trp_res['snr'], 2),
                'tht_snr': round(tht_res['snr'], 2),
                'trp_amp': round(trp_res['amplitude'], 2),
                'tht_amp': round(tht_res['amplitude'], 2),
            })

    result = pd.DataFrame(rows)
    order = pd.Categorical(result['sample'], categories=SAMPLES, ordered=True)
    result = result.assign(_order=order).sort_values(['_order', 'replicate']).drop(columns=['_order'])
    summary = result.groupby('sample', sort=False)['combined_lag_min'].agg(['mean', 'std']).reset_index()
    summary.columns = ['sample', 'combined_mean_min', 'combined_sd_min']
    summary['combined_mean_min'] = summary['combined_mean_min'].round(2)
    summary['combined_sd_min'] = summary['combined_sd_min'].round(2)
    outliers = detect_outliers(result)

    with pd.ExcelWriter(OUT_XLSX) as writer:
        result.to_excel(writer, index=False, sheet_name='replicate_lag')
        summary.to_excel(writer, index=False, sheet_name='summary')
        outliers.to_excel(writer, index=False, sheet_name='outliers')
    result.to_csv(OUT_CSV, index=False)

    print('Saved:', OUT_XLSX)
    print('Saved:', OUT_CSV)
    print('--- SUMMARY ---')
    print(summary.to_string(index=False))
    print('--- OUTLIERS ---')
    if len(outliers):
        print(outliers.to_string(index=False))
    else:
        print('None')


if __name__ == '__main__':
    main()

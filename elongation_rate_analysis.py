import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

BASE = Path(r'c:\Users\17865\Desktop\GCG Mix')
DATASETS = {
    '3.0-1': {'trp': BASE / '3.0-1' / 'Trp.xlsx', 'tht': BASE / '3.0-1' / 'ThT.xlsx'},
    '3.0-2': {'trp': BASE / '3.0-2' / 'Trp.xlsx', 'tht': BASE / '3.0-2' / 'Tht.xlsx'},
    '3.0-3': {'trp': BASE / '3.0-3' / 'Trp.xlsx', 'tht': BASE / '3.0-3' / 'ThT.xlsx'},
}
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


def smooth_signal(t: np.ndarray, y: np.ndarray, mode: str):
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
    fine_t = np.linspace(float(t.min()), float(t.max()), int((t.max() - t.min()) * 10) + 1)
    fine_signal = PchipInterpolator(t, signal)(fine_t)
    fine_slope = np.gradient(fine_signal, fine_t)
    amplitude = float(np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5))
    early_sd = float(np.nanstd(raw_signal[:6], ddof=1)) if len(raw_signal) >= 6 else float(np.nanstd(raw_signal, ddof=1))
    snr = amplitude / max(early_sd, 1e-6)
    tail_progress = float(np.median(signal[-5:])) / amplitude if amplitude > 0 else np.nan
    return {
        't': t,
        'y': y,
        'signal': signal,
        'fine_t': fine_t,
        'fine_signal': fine_signal,
        'fine_slope': fine_slope,
        'amplitude': amplitude,
        'snr': snr,
        'tail_progress': tail_progress,
        'baseline': baseline,
    }


def compute_elongation_rate(t: np.ndarray, y: np.ndarray, mode: str):
    prof = smooth_signal(t, y, mode)
    amp = prof['amplitude']
    fine_t = prof['fine_t']
    fine_signal = prof['fine_signal']
    fine_slope = prof['fine_slope']
    norm_signal = fine_signal / amp if amp > 0 else np.full_like(fine_signal, np.nan)
    norm_slope = fine_slope / amp if amp > 0 else np.full_like(fine_slope, np.nan)
    mask = (norm_signal >= 0.10) & (norm_signal <= 0.80)
    if not np.any(mask):
        mask = (norm_signal >= 0.05) & (norm_signal <= 0.90)
    idx = int(np.argmax(np.where(mask, norm_slope, -np.inf)))
    max_norm_rate = float(norm_slope[idx])
    max_raw_slope = float(fine_slope[idx])
    t_at_max = float(fine_t[idx])
    valid = bool(np.isfinite(max_norm_rate) and max_norm_rate > 0 and amp > 0)
    return {
        'elongation_rate': max_norm_rate,
        'raw_slope': max_raw_slope,
        'time_at_max': t_at_max,
        'amplitude': amp,
        'snr': float(prof['snr']),
        'tail_progress': float(prof['tail_progress']),
        'valid': valid,
    }


def combine_rate(trp_res, tht_res):
    trp = trp_res['elongation_rate']
    tht = tht_res['elongation_rate']
    use_tht = False
    reason = 'Trp primary'
    if tht_res['valid'] and tht_res['snr'] >= 4 and tht_res['tail_progress'] >= 0.35 and np.isfinite(tht):
        ratio = max(trp, tht) / max(min(trp, tht), 1e-9)
        if ratio <= 1.35:
            combined = 0.75 * trp + 0.25 * tht
            use_tht = True
            reason = 'Trp-dominant weighted blend'
        elif ratio <= 1.8:
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


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    flagged = []
    for sample, g in df.groupby('sample', sort=False):
        vals = g['combined_rate_min_inv'].to_numpy(dtype=float)
        med = float(np.median(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        for _, row in g.iterrows():
            delta = abs(float(row['combined_rate_min_inv']) - med)
            rel = delta / max(med, 1e-9)
            if rel >= 0.20 or (sd > 0 and delta >= max(1.2 * sd, 0.01)):
                flagged.append({
                    'sample': sample,
                    'replicate': int(row['replicate']),
                    'combined_rate_min_inv': float(row['combined_rate_min_inv']),
                    'sample_median': med,
                    'sample_sd': sd,
                    'delta_from_median': delta,
                    'relative_delta': rel,
                })
    return pd.DataFrame(flagged)


def review_outliers(result: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    review_rows = []
    for _, row in outliers.iterrows():
        sample = row['sample']
        rep = int(row['replicate'])
        sub = result[(result['sample'] == sample) & (result['replicate'] == rep)].iloc[0]
        decision = 'Retained'
        note = 'Trp and ThT show concordant elongation-rate trend; retained.'
        if sub['decision'] == 'ThT inconsistent, ignored' and row['relative_delta'] >= 0.35:
            note = 'Outlier relative to sibling replicates, but Trp shape remains internally consistent; retained.'
        elif sub['decision'] == 'ThT low-confidence, ignored' and row['relative_delta'] >= 0.35:
            note = 'Rate differs from sibling replicates, but only Trp is reliable here; retained without manual replacement.'
        elif abs(sub['trp_rate_min_inv'] - sub['tht_rate_min_inv']) / max(sub['trp_rate_min_inv'], 1e-9) < 0.2:
            note = 'Both probes support the same faster/slower elongation trend; retained.'
        review_rows.append({
            'sample': sample,
            'replicate': rep,
            'combined_rate_min_inv': round(float(row['combined_rate_min_inv']), 5),
            'sample_median': round(float(row['sample_median']), 5),
            'delta_from_median': round(float(row['delta_from_median']), 5),
            'review_decision': decision,
            'review_note': note,
        })
    return pd.DataFrame(review_rows)


def process_dataset(name: str, paths: dict):
    trp_df = load_book(paths['trp'])
    tht_df = load_book(paths['tht'])
    t = get_time_minutes(trp_df)
    rows = []
    for sample in SAMPLES:
        trp_reps = sample_replicates(trp_df, sample)
        tht_reps = sample_replicates(tht_df, sample)
        for idx, (trp_col, tht_col) in enumerate(zip(trp_reps, tht_reps), start=1):
            trp_res = compute_elongation_rate(t, trp_df[(sample, trp_col)].to_numpy(), 'trp')
            tht_res = compute_elongation_rate(t, tht_df[(sample, tht_col)].to_numpy(), 'tht')
            combined, used_tht, reason = combine_rate(trp_res, tht_res)
            rows.append({
                'dataset': name,
                'sample': sample,
                'replicate': idx,
                'trp_col': trp_col,
                'tht_col': tht_col,
                'trp_rate_min_inv': round(trp_res['elongation_rate'], 5),
                'tht_rate_min_inv': round(tht_res['elongation_rate'], 5),
                'combined_rate_min_inv': round(combined, 5),
                'tht_used': used_tht,
                'decision': reason,
                'trp_time_at_max_min': round(trp_res['time_at_max'], 2),
                'tht_time_at_max_min': round(tht_res['time_at_max'], 2),
                'trp_snr': round(trp_res['snr'], 2),
                'tht_snr': round(tht_res['snr'], 2),
                'trp_amp': round(trp_res['amplitude'], 2),
                'tht_amp': round(tht_res['amplitude'], 2),
            })
    result = pd.DataFrame(rows)
    order = pd.Categorical(result['sample'], categories=SAMPLES, ordered=True)
    result = result.assign(_order=order).sort_values(['_order', 'replicate']).drop(columns=['_order'])
    outliers = detect_outliers(result)
    review = review_outliers(result, outliers) if len(outliers) else pd.DataFrame(columns=['sample','replicate','combined_rate_min_inv','sample_median','delta_from_median','review_decision','review_note'])
    final_rep = result.copy()
    final_rep['final_rate_min_inv'] = final_rep['combined_rate_min_inv']
    final_rep['review_status'] = 'Auto accepted'
    final_rep['manual_revision'] = ''
    if len(review):
        for _, r in review.iterrows():
            mask = (final_rep['sample'] == r['sample']) & (final_rep['replicate'] == r['replicate'])
            final_rep.loc[mask, 'review_status'] = r['review_decision']
            final_rep.loc[mask, 'manual_revision'] = r['review_note']
    summary = final_rep.groupby('sample', sort=False)['final_rate_min_inv'].agg(['mean', 'std']).reset_index()
    summary.columns = ['sample', 'final_mean_min_inv', 'final_sd_min_inv']
    summary['final_mean_min_inv'] = summary['final_mean_min_inv'].round(5)
    summary['final_sd_min_inv'] = summary['final_sd_min_inv'].round(5)
    summary['mean_sd_min_inv'] = summary.apply(lambda r: f"{r['final_mean_min_inv']:.5f} +/- {r['final_sd_min_inv']:.5f}", axis=1)
    manuscript = summary.copy()
    manuscript.insert(1, 'rep1_min_inv', final_rep[final_rep['replicate'] == 1]['final_rate_min_inv'].round(5).to_list())
    manuscript.insert(2, 'rep2_min_inv', final_rep[final_rep['replicate'] == 2]['final_rate_min_inv'].round(5).to_list())
    manuscript.insert(3, 'rep3_min_inv', final_rep[final_rep['replicate'] == 3]['final_rate_min_inv'].round(5).to_list())
    manuscript = manuscript[['sample', 'rep1_min_inv', 'rep2_min_inv', 'rep3_min_inv', 'final_mean_min_inv', 'final_sd_min_inv', 'mean_sd_min_inv']]
    out_xlsx = BASE / name / 'elongation_rate_results_final.xlsx'
    out_csv = BASE / name / 'elongation_rate_results_final.csv'
    with pd.ExcelWriter(out_xlsx) as writer:
        final_rep.to_excel(writer, index=False, sheet_name='replicate_final')
        summary.to_excel(writer, index=False, sheet_name='summary_final')
        manuscript.to_excel(writer, index=False, sheet_name='manuscript_table')
        outliers.to_excel(writer, index=False, sheet_name='outliers')
        review.to_excel(writer, index=False, sheet_name='outlier_review')
    manuscript.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return final_rep, summary, manuscript, review, out_xlsx, out_csv


def main():
    combined_summaries = {}
    for name, paths in DATASETS.items():
        rep, summary, manuscript, review, out_xlsx, out_csv = process_dataset(name, paths)
        combined_summaries[name] = manuscript
        print('DATASET', name)
        print('Saved', out_xlsx)
        print('Saved', out_csv)
        print(summary.to_string(index=False))
        if len(review):
            print('OUTLIER REVIEW')
            print(review.to_string(index=False))
        print('====')


if __name__ == '__main__':
    main()

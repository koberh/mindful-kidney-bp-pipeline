#!/usr/bin/env python3
"""
Generate participant-facing biometric report matching the IRB Appendix 7a mockup.

Pages (official):
  1. Cover page
  2. Calm Health App Usage & Mindfulness Practice
  3. Home Blood Pressure (weekly, from cuff CSV)
  4. Sleep & Physical Activity (weekly sleep + steps)
  5. Heart Rate & HRV (nightly averages by week)
  6. Multi-Modal Timeline Overview (heatmap grid)
  7. Continuous BP Trend + AHA categorical staging
  8. Data Sources

Pages (informal daily snapshots, appended after Data Sources):
  9+. Weekly Snapshot — one page per study week

Usage:
  python generate_full_report.py \
      --subject-id   SUBJ01 \
      --inference-dir "D:/Mindful Kidney/Biometric Report/inference" \
      --data-dir      "D:/Mindful Kidney/Biometric Report/data/OneDrive_actigraph" \
      --cuff-csv      "D:/Mindful Kidney/Biometric Report/data/centrepoint_mktest01/cuff_readings.csv" \
      --out-dir       "D:/Mindful Kidney/Biometric Report/output" \
      --name          "MK02" \
      --timezone      US/Eastern \
      --study-start   2026-04-28
"""

import argparse
import json
import glob
import csv as csv_mod
from pathlib import Path
from datetime import timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

HRV_RELIABILITY_MIN = 70.0
N_WEEKS = 6

SBP_COLOR         = '#C62828'
DBP_COLOR         = '#1565C0'
HR_COLOR          = '#E65100'
RMSSD_COLOR       = '#2E7D32'
SLEEP_COLOR       = '#5C6BC0'
STEP_COLOR        = '#00838F'
GRID_COLOR        = '#EEEEEE'
BG                = '#FAFAFA'
PLACEHOLDER_BG    = '#F5F5F5'
PLACEHOLDER_TEXT  = '#9E9E9E'

WEEK_LABELS = [f'Wk {i}' for i in range(1, N_WEEKS + 1)]

# AHA 2017 stage colors
AHA_COLORS = {
    'Normal':   '#43A047',
    'Elevated': '#8BC34A',
    'Stage 1':  '#FFC107',
    'Stage 2':  '#FF5722',
    'Crisis':   '#B71C1C',
}


def classify_aha_stage(sbp, dbp):
    """Return (label, color_hex) per AHA 2017 thresholds."""
    if np.isnan(sbp) or np.isnan(dbp):
        return '—', PLACEHOLDER_TEXT
    if sbp >= 180 or dbp >= 120:
        return 'Crisis',   AHA_COLORS['Crisis']
    if sbp >= 140 or dbp >= 90:
        return 'Stage 2',  AHA_COLORS['Stage 2']
    if sbp >= 130 or dbp >= 80:
        return 'Stage 1',  AHA_COLORS['Stage 1']
    if sbp >= 120 and dbp < 80:
        return 'Elevated', AHA_COLORS['Elevated']
    return 'Normal',   AHA_COLORS['Normal']


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--subject-id',    required=True)
    p.add_argument('--inference-dir', required=True)
    p.add_argument('--data-dir',      required=True)
    p.add_argument('--cuff-csv',      default=None)
    p.add_argument('--out-dir',       default=None)
    p.add_argument('--name',          default=None)
    p.add_argument('--timezone',      default='US/Eastern')
    p.add_argument('--study-start',   default=None)
    p.add_argument('--calm-data-dir', default=None)
    p.add_argument('--calm-member-id', default=None)
    p.add_argument('--calm-study-start', default=None)
    p.add_argument('--leap-dir', default=None)
    p.add_argument('--p5-pred-csv', default=None,
                   help='Path to Phase 5 LSTM predictions CSV for cuff-vs-model comparison page.')
    p.add_argument('--p3-pred-csv', default=None,
                   help='Path to Phase 3 LSTM predictions CSV for cuff-vs-model comparison page.')
    return p.parse_args()


def to_local(series_utc, tz):
    return pd.to_datetime(series_utc, utc=True, format='mixed').dt.tz_convert(tz)


def assign_week(dates_series, study_start):
    delta = (pd.to_datetime(dates_series).normalize() -
             pd.Timestamp(study_start).normalize()).days
    if hasattr(delta, '__iter__'):
        return pd.Series(delta).apply(lambda d: int(np.clip(d // 7 + 1, 1, N_WEEKS)))
    return int(np.clip(delta // 7 + 1, 1, N_WEEKS))


def assign_week_series(dates_series, study_start):
    ts = pd.to_datetime(dates_series)
    start = pd.Timestamp(study_start).normalize()
    delta_days = (ts.dt.normalize() - start).dt.days
    return (delta_days // 7 + 1).clip(1, N_WEEKS)


def page_header(fig, title, subtitle=''):
    fig.text(0.5, 0.97, title, ha='center', va='top',
             fontsize=13, fontweight='bold', color='#212121')
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha='center', va='top',
                 fontsize=9.5, color='#616161')


def placeholder_panel(ax, label, note='Data not yet available for this section.'):
    ax.set_facecolor(PLACEHOLDER_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
    ax.text(0.5, 0.58, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=11,
            fontweight='bold', color=PLACEHOLDER_TEXT)
    ax.text(0.5, 0.38, note, transform=ax.transAxes,
            ha='center', va='center', fontsize=8.5,
            color=PLACEHOLDER_TEXT, style='italic')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sleep_windows_raw(inf_dir, subject_id):
    p = Path(inf_dir) / subject_id / 'sleep_windows.json'
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f).get('nights', [])


def load_sleep_windows_from_leap(leap_dir, subject_id, tz):
    fpath = Path(leap_dir) / 'subjectsleepperiodmetrics.csv'
    if not fpath.exists():
        return []
    df = pd.read_csv(fpath)
    df = df[df['Subject'] == subject_id].copy()
    if df.empty:
        return []
    df['in_utc']  = pd.to_datetime(df['InBedTime']).dt.tz_localize('UTC')
    df['out_utc'] = pd.to_datetime(df['OutBedTime']).dt.tz_localize('UTC')
    df['date_local'] = df['in_utc'].dt.tz_convert(tz).dt.date.astype(str)
    df['sleep_min']  = pd.to_numeric(df['TimeAsleepInMinutes'], errors='coerce').fillna(0)
    nights = []
    for date_local, grp in df.groupby('date_local', sort=True):
        total_min = grp['sleep_min'].sum()
        start_ms  = int(grp['in_utc'].min().value  // 1_000_000)
        end_ms    = int(grp['out_utc'].max().value  // 1_000_000)
        nights.append({
            'date':               date_local,
            'sleep_start_utc_ms': start_ms,
            'sleep_end_utc_ms':   end_ms,
            'duration_hours':     round(float(total_min) / 60, 2),
            'source':             'actigraph_leap',
        })
    return nights


def load_steps_weekly(leap_dir, subject_id, study_start):
    fpath = Path(leap_dir) / 'subjectdaystats.csv'
    if not fpath.exists():
        return {}
    df = pd.read_csv(fpath)
    df = df[df['Subject'] == subject_id].copy()
    if df.empty:
        return {}
    df['date']  = pd.to_datetime(df['Date']).dt.date.astype(str)
    df['steps'] = pd.to_numeric(df['WearFilteredSteps'], errors='coerce')
    df['week']  = assign_week_series(df['date'], study_start)
    return df.groupby('week')['steps'].mean().round(0).to_dict()


def load_steps_daily(leap_dir, subject_id):
    """Return {date_str: steps} from subjectdaystats.csv."""
    fpath = Path(leap_dir) / 'subjectdaystats.csv'
    if not fpath.exists():
        return {}
    df = pd.read_csv(fpath)
    df = df[df['Subject'] == subject_id].copy()
    if df.empty:
        return {}
    df['date']  = pd.to_datetime(df['Date']).dt.date.astype(str)
    df['steps'] = pd.to_numeric(df['WearFilteredSteps'], errors='coerce')
    return df.dropna(subset=['steps']).set_index('date')['steps'].astype(int).to_dict()


def load_sleep_weekly(sleep_windows_raw, study_start):
    if not sleep_windows_raw:
        return {}
    rows = [{'date': n['date'], 'hours': n['duration_hours']}
            for n in sleep_windows_raw]
    df = pd.DataFrame(rows)
    df['week'] = assign_week_series(df['date'], study_start)
    return df.groupby('week')['hours'].mean().to_dict()


def load_calm_weekly(calm_dir, member_id, study_start):
    seen = set()
    mind_counts = {}
    other_counts = {}
    for wd in sorted(glob.glob(str(Path(calm_dir) / 'week_*'))):
        for fpath in glob.glob(str(Path(wd) / '*sessions*')):
            with open(fpath, newline='', encoding='utf-8') as fh:
                for row in csv_mod.DictReader(fh):
                    if row.get('partner_member_id', '').lower() != member_id.lower():
                        continue
                    sid = row.get('user_session_id', '')
                    if sid in seen:
                        continue
                    seen.add(sid)
                    started = row.get('session_started_at', '')
                    if not started:
                        continue
                    try:
                        dt = pd.to_datetime(started[:10]).date()
                    except Exception:
                        continue
                    delta = (dt - pd.Timestamp(study_start).date()).days
                    week  = int(np.clip(delta // 7 + 1, 1, N_WEEKS))
                    prog_type = row.get('media_program_type', '').lower()
                    if prog_type in ('music', 'soundscape', 'sleep'):
                        other_counts[week] = other_counts.get(week, 0) + 1
                    else:
                        mind_counts[week] = mind_counts.get(week, 0) + 1
    return mind_counts, other_counts


def load_calm_daily(calm_dir, member_id):
    """Return {date_str: session_count} across all weekly exports."""
    seen = set()
    counts = {}
    for wd in sorted(glob.glob(str(Path(calm_dir) / 'week_*'))):
        for fpath in glob.glob(str(Path(wd) / '*sessions*')):
            with open(fpath, newline='', encoding='utf-8') as fh:
                for row in csv_mod.DictReader(fh):
                    if row.get('partner_member_id', '').lower() != member_id.lower():
                        continue
                    sid = row.get('user_session_id', '')
                    if sid in seen:
                        continue
                    seen.add(sid)
                    started = row.get('session_started_at', '')
                    if not started:
                        continue
                    try:
                        date = str(pd.to_datetime(started[:10]).date())
                    except Exception:
                        continue
                    counts[date] = counts.get(date, 0) + 1
    return counts


def load_cuff_bp_weekly(cuff_csv, subject_id, study_start, tz):
    if not cuff_csv or not Path(cuff_csv).exists():
        return {}, set()
    df = pd.read_csv(cuff_csv)
    id_col = 'subject_id' if 'subject_id' in df.columns else 'participant_id'
    df = df[df[id_col] == subject_id].copy()
    if df.empty:
        return {}, set()
    df['t'] = to_local(df['timestamp_utc'], tz)
    df['date'] = df['t'].dt.date.astype(str)
    df['week'] = assign_week_series(df['date'], study_start)
    sbp_by_week = df.groupby('week')['sbp_mmhg'].mean().to_dict()
    dbp_by_week = df.groupby('week')['dbp_mmhg'].mean().to_dict()
    reviewed = set()
    if 'physician_reviewed' in df.columns:
        mask = df['physician_reviewed'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
        reviewed = set(df.loc[mask, 'week'].tolist())
    return {'sbp': sbp_by_week, 'dbp': dbp_by_week}, reviewed


def load_hr_hrv_weekly(data_dir, sleep_windows_raw, study_start, tz):
    hr_path = Path(data_dir) / 'HeartRate.csv'
    try:
        hr_df = pd.read_csv(hr_path, on_bad_lines='skip')
        if 'timestamp' not in hr_df.columns or 'HeartRate' not in hr_df.columns:
            print(f'Warning: {hr_path} missing expected columns (has: {list(hr_df.columns)[:4]}...) — skipping HR data.')
            hr_df = pd.DataFrame(columns=['t', 'hr'])
        else:
            hr_df['t']  = to_local(hr_df['timestamp'], tz)
            hr_df['hr'] = pd.to_numeric(hr_df['HeartRate'], errors='coerce')
    except Exception as e:
        print(f'Warning: could not load {hr_path}: {e} — skipping HR data.')
        hr_df = pd.DataFrame(columns=['t', 'hr'])

    hrv_path = Path(data_dir) / 'HeartRateVar.csv'
    try:
        hrv_df = pd.read_csv(hrv_path, on_bad_lines='skip', encoding='latin-1')
        if 'start' not in hrv_df.columns or 'rmssd' not in hrv_df.columns:
            print(f'Warning: {hrv_path} missing expected columns — skipping HRV data.')
            hrv_df = pd.DataFrame(columns=['t', 'rmssd'])
        else:
            hrv_df['t'] = to_local(hrv_df['start'], tz)
            hrv_df = hrv_df[hrv_df['hrv_reliability'] >= HRV_RELIABILITY_MIN].copy()
            hrv_df['rmssd'] = pd.to_numeric(hrv_df['rmssd'], errors='coerce')
    except Exception as e:
        print(f'Warning: could not load {hrv_path}: {e} — skipping HRV data.')
        hrv_df = pd.DataFrame(columns=['t', 'rmssd'])

    study_window_start = pd.Timestamp(study_start).tz_localize(tz)
    study_window_end   = study_window_start + pd.Timedelta(weeks=N_WEEKS)
    if not hr_df.empty:
        hr_df  = hr_df[ hr_df['t'].between(study_window_start, study_window_end)].copy()
    if not hrv_df.empty:
        hrv_df = hrv_df[hrv_df['t'].between(study_window_start, study_window_end)].copy()

    nightly_hr, nightly_hrv = {}, {}
    for night in sleep_windows_raw:
        date = night['date']
        start = pd.to_datetime(night['sleep_start_utc_ms'], unit='ms', utc=True).tz_convert(tz)
        end   = pd.to_datetime(night['sleep_end_utc_ms'],   unit='ms', utc=True).tz_convert(tz)
        if not hr_df.empty:
            hr_sleep = hr_df.loc[hr_df['t'].between(start, end), 'hr'].dropna()
            if len(hr_sleep) >= 5:
                nightly_hr[date] = hr_sleep.median()
        if not hrv_df.empty:
            hrv_sleep = hrv_df.loc[hrv_df['t'].between(start, end), 'rmssd'].dropna()
            if len(hrv_sleep) >= 3:
                nightly_hrv[date] = hrv_sleep.median()

    if not nightly_hr and not hr_df.empty:
        hr_df['date'] = hr_df['t'].dt.date.astype(str)
        nightly_hr = hr_df.groupby('date')['hr'].median().to_dict()
    if not nightly_hrv and not hrv_df.empty:
        hrv_df['date'] = hrv_df['t'].dt.date.astype(str)
        nightly_hrv = hrv_df.groupby('date')['rmssd'].median().to_dict()

    def to_weekly(daily_dict):
        df = pd.DataFrame(list(daily_dict.items()), columns=['date', 'value'])
        df['week'] = assign_week_series(df['date'], study_start)
        return df.groupby('week')['value'].mean().to_dict()

    return to_weekly(nightly_hr), to_weekly(nightly_hrv), nightly_hr, nightly_hrv


def load_bp_trend_data(inf_dir, subject_id, tz):
    for fname in ('bp_trend_calibrated.csv', 'bp_trend.csv'):
        p = Path(inf_dir) / subject_id / fname
        if p.exists():
            break
    else:
        return None, None
    df = pd.read_csv(p)
    if 'datetime_utc' not in df.columns and 'window_start_utc' in df.columns:
        df = df.rename(columns={'window_start_utc': 'datetime_utc'})
    if 'is_sleep' not in df.columns and 'sleep_flag' in df.columns:
        df = df.rename(columns={'sleep_flag': 'is_sleep'})
    df['t'] = to_local(df['datetime_utc'], tz)
    sbp_col = next((c for c in ('sbp_calibrated', 'sbp_cal', 'sbp_mmhg') if c in df.columns), None)
    dbp_col = next((c for c in ('dbp_calibrated', 'dbp_cal', 'dbp_mmhg') if c in df.columns), None)
    df['sbp'] = pd.to_numeric(df[sbp_col], errors='coerce')
    df['dbp'] = pd.to_numeric(df[dbp_col], errors='coerce')
    df = df.dropna(subset=['sbp', 'dbp'])

    dip = None
    for dname in ('nocturnal_dip_calibrated.json', 'nocturnal_dip.json'):
        dp = Path(inf_dir) / subject_id / dname
        if dp.exists():
            with open(dp) as f:
                dip = json.load(f)
            break
    return df, dip


def compute_daily_bp_stage(bp_df, tz):
    """Return {date_str: (stage, color, sbp_med, dbp_med)} using AHA 2017."""
    if bp_df is None or bp_df.empty:
        return {}
    df = bp_df.copy()
    df['date'] = df['t'].dt.date.astype(str)
    daily = df.groupby('date')[['sbp', 'dbp']].median()
    result = {}
    for date, row in daily.iterrows():
        stage, color = classify_aha_stage(row['sbp'], row['dbp'])
        result[date] = (stage, color, round(float(row['sbp']), 1), round(float(row['dbp']), 1))
    return result


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_cover(pdf, name, subject_id, study_period, report_date, bp_reviewer):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)

    meta = [
        ('Participant ID',   subject_id),
        ('Study Period',     study_period),
        ('Report Generated', report_date),
        ('BP Reviewed By',   bp_reviewer or 'Pending'),
    ]
    col_x = [0.08, 0.30, 0.57, 0.78]
    for (label, val), x in zip(meta, col_x):
        fig.text(x, 0.90, label, fontsize=8,  color='#757575', ha='left', va='top')
        fig.text(x, 0.86, val,   fontsize=9.5, fontweight='bold',
                 color='#212121', ha='left', va='top')

    for ax_rect, ypos in [([0.06, 0.845, 0.88, 0.002], 0.845)]:
        divider = fig.add_axes(ax_rect)
        divider.set_facecolor('#BDBDBD')
        divider.axis('off')

    fig.text(0.5, 0.79, 'Biometric Report', ha='center',
             fontsize=26, fontweight='bold', color='#212121')
    fig.text(0.5, 0.725, 'Mindful Kidney Study', ha='center',
             fontsize=14, color='#1565C0')
    fig.text(0.5, 0.675, name or subject_id, ha='center',
             fontsize=16, color='#424242')

    divider2 = fig.add_axes([0.2, 0.645, 0.6, 0.002])
    divider2.set_facecolor('#BDBDBD')
    divider2.axis('off')

    sections = [
        ('1 · Calm Health App Usage',
         'Guided mindfulness minutes per week vs. 45-minute weekly goal.'),
        ('2 · Mindfulness Practice',
         'App-guided sessions vs. self-directed practice captured by event markers.'),
        ('3 · Home Blood Pressure',
         'Weekly average systolic and diastolic BP from home cuff readings (weeks 5-6).'),
        ('4 · Sleep & Physical Activity',
         'Average nightly sleep duration and daily step count by week.'),
        ('5 · Heart Rate & Heart Rate Variability',
         'Nightly resting HR and RMSSD by week. Lower HR and higher RMSSD reflect better recovery.'),
        ('6 · Multi-Modal Timeline Overview',
         'All data streams side by side -- spot patterns across your 6-week journey.'),
        ('7 · BP Trend + AHA Staging',
         'Continuous wristband BP estimate with AHA 2017 categorical staging zones.'),
    ]
    for i, (title, desc) in enumerate(sections):
        y = 0.600 - i * 0.072
        fig.text(0.14, y,        title, fontsize=10, fontweight='bold',
                 color='#212121', va='top')
        fig.text(0.14, y - 0.025, desc, fontsize=8.5, color='#616161', va='top')

    fig.text(0.5, 0.04,
             'These measurements come from your ActiGraph LEAP wristband and home blood pressure monitor.\n'
             'They are research estimates -- not a substitute for clinical evaluation. '
             'Please bring this report to your care team.',
             ha='center', fontsize=8.5, color='#9E9E9E', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_calm_health(pdf, name, calm_mindfulness=None, calm_other=None, selfdirected=None,
                     count_mode=False):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig,
                f'App Usage & Mindfulness Practice -- {name or "Participant"}',
                'Weeks 1-6')
    gs = fig.add_gridspec(2, 1, hspace=0.5, top=0.85, bottom=0.08,
                          left=0.1, right=0.92)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('  Mindfulness exercises      Sleep & other content     Goal (45 min/week)',
                  fontsize=9, color='#616161', pad=6)
    if calm_mindfulness:
        x          = np.arange(1, N_WEEKS + 1)
        mind_vals  = [calm_mindfulness.get(w, 0) for w in range(1, N_WEEKS + 1)]
        other_vals = [calm_other.get(w, 0) if calm_other else 0
                      for w in range(1, N_WEEKS + 1)]
        ax1.bar(x, mind_vals,  width=0.55, color='#7E57C2',
                label='Mindfulness exercises', zorder=3)
        ax1.bar(x, other_vals, width=0.55, bottom=mind_vals,
                color='#CE93D8', label='Sleep & other content', zorder=3)
        if not count_mode:
            ax1.axhline(45, color='#424242', linewidth=1.2, linestyle='--', zorder=2)
            ax1.text(N_WEEKS + 0.15, 46.5, 'Goal (45 min)',
                     fontsize=7.5, color='#616161', va='bottom')
        max_total = max((mv + ov for mv, ov in zip(mind_vals, other_vals)), default=1)
        for w_i, (mv, ov) in enumerate(zip(mind_vals, other_vals)):
            total = mv + ov
            if total > 0:
                ax1.text(w_i + 1, total + max_total * 0.03, f'{total:.0f}',
                         ha='center', va='bottom', fontsize=9,
                         fontweight='bold', color='#424242')
        ax1.set_xticks(x)
        ax1.set_xticklabels(WEEK_LABELS, fontsize=10)
        ax1.set_ylabel('Sessions' if count_mode else 'Minutes', fontsize=10)
        ax1.set_xlim(0.4, N_WEEKS + 0.6)
        ax1.set_ylim(0, max_total * 1.25)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax1.set_facecolor('#FFFFFF')
        ax1.set_title('1 · Calm Health App Usage', fontsize=10, color='#424242', pad=6)
        if count_mode:
            ax1.set_title('1 · Calm Health App Usage  (session count -- duration not in export)',
                          fontsize=9, color='#424242', pad=6)
    else:
        placeholder_panel(ax1, '1 · Calm Health App Usage',
                          'Guided mindfulness minutes per week will appear here\n'
                          'once Calm Health session data is available.')

    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('  Calm Health (guided)      Self-directed (Actigraph event marker)',
                  fontsize=9, color='#616161', pad=6)
    if calm_mindfulness and selfdirected:
        x     = np.arange(1, N_WEEKS + 1)
        width = 0.35
        guided_vals = [(calm_mindfulness.get(w, 0) + (calm_other.get(w, 0) if calm_other else 0))
                       for w in range(1, N_WEEKS + 1)]
        sdir_vals   = [selfdirected.get(w, 0) for w in range(1, N_WEEKS + 1)]
        ax2.bar(x - width / 2, guided_vals, width=width, color='#7E57C2',
                label='Calm Health (guided)', zorder=3)
        ax2.bar(x + width / 2, sdir_vals,   width=width, color='#AB47BC',
                label='Self-directed', zorder=3)
        for xi, gv, sv in zip(x, guided_vals, sdir_vals):
            if gv > 0:
                ax2.text(xi - width / 2, gv + 0.8, f'{gv:.0f}',
                         ha='center', va='bottom', fontsize=8.5,
                         color='#7E57C2', fontweight='bold')
            if sv > 0:
                ax2.text(xi + width / 2, sv + 0.8, f'{sv:.0f}',
                         ha='center', va='bottom', fontsize=8.5,
                         color='#AB47BC', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(WEEK_LABELS, fontsize=10)
        ax2.set_ylabel('Minutes', fontsize=10)
        ax2.set_xlim(0.4, N_WEEKS + 0.6)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax2.set_facecolor('#FFFFFF')
        ax2.set_title('2 · Mindfulness Practice -- App-Guided vs. Self-Directed',
                      fontsize=10, color='#424242', pad=6)
        fig.text(0.5, 0.04,
                 'Self-directed sessions: intervals between event marker presses '
                 'not overlapping with Calm Health session timestamps. '
                 'Sessions under 2 minutes excluded.',
                 ha='center', fontsize=8, color='#9E9E9E', style='italic')
    else:
        placeholder_panel(ax2, '2 · Mindfulness Practice -- App-Guided vs. Self-Directed',
                          'Calm Health session timestamps and Actigraph event marker data\n'
                          'will appear here once both sources are available.')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_home_bp(pdf, bp_data, reviewed_weeks, name, study_period):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig,
                f'3 · Home Blood Pressure -- {name or "Participant"}',
                f'{study_period}  ·  Home cuff readings  ·  * = Physician reviewed')

    ax = fig.add_axes([0.12, 0.20, 0.76, 0.60])
    sbp = bp_data.get('sbp', {})
    dbp = bp_data.get('dbp', {})

    if not sbp:
        placeholder_panel(ax, '3 · Home Blood Pressure',
                          'Home cuff readings (weeks 5-6) will appear here.')
        fig.text(0.5, 0.08,
                 'Per protocol, home blood pressure readings are collected during weeks 5-6.',
                 ha='center', fontsize=8.5, color='#9E9E9E', style='italic')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return

    x = np.arange(1, N_WEEKS + 1)
    sbp_vals = [sbp.get(w, np.nan) for w in range(1, N_WEEKS + 1)]
    dbp_vals = [dbp.get(w, np.nan) for w in range(1, N_WEEKS + 1)]

    for w in range(1, N_WEEKS + 1):
        if w not in sbp:
            ax.axvspan(w - 0.45, w + 0.45, color='#F9F9F9', zorder=1)

    valid_w   = [w for w in range(1, N_WEEKS + 1) if w in sbp]
    valid_sbp = [sbp[w] for w in valid_w]
    valid_dbp = [dbp.get(w, np.nan) for w in valid_w]

    ax.plot(valid_w, valid_sbp, 'o-', color=SBP_COLOR, linewidth=2.2,
            markersize=8, zorder=4, label='Systolic (SBP)')
    ax.plot(valid_w, valid_dbp, 's-', color=DBP_COLOR, linewidth=2.2,
            markersize=8, zorder=4, label='Diastolic (DBP)')

    for w in reviewed_weeks:
        if w in sbp:
            ax.plot(w, sbp[w] + 14, '*', color='#F9A825',
                    markersize=15, zorder=5, clip_on=False)

    for w, s, d in zip(valid_w, valid_sbp, valid_dbp):
        ax.text(w, s + 2.5, f'{s:.0f}', ha='center', va='bottom',
                fontsize=9, color=SBP_COLOR, fontweight='bold')
        if not np.isnan(d):
            ax.text(w, d - 4.5, f'{d:.0f}', ha='center', va='top',
                    fontsize=9, color=DBP_COLOR, fontweight='bold')

    ax.axhline(120, color='#BDBDBD', linewidth=0.9, linestyle='--', zorder=2)
    ax.axhline(80,  color='#BDBDBD', linewidth=0.9, linestyle='--', zorder=2)
    ax.text(N_WEEKS + 0.12, 120.5, 'Normal SBP (120)',
            fontsize=7.5, color='#9E9E9E', va='bottom')
    ax.text(N_WEEKS + 0.12, 80.5,  'Normal DBP (80)',
            fontsize=7.5, color='#9E9E9E', va='bottom')

    all_vals = [v for v in sbp_vals + dbp_vals if not np.isnan(v)]
    ymin = max(50,  min(all_vals) - 18) if all_vals else 60
    ymax = min(200, max(all_vals) + 22) if all_vals else 160

    ax.set_xlim(0.4, N_WEEKS + 0.6)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(1, N_WEEKS + 1))
    ax.set_xticklabels(WEEK_LABELS, fontsize=10)
    ax.set_ylabel('Blood Pressure (mmHg)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_facecolor('#FFFFFF')

    handles = [
        plt.Line2D([0], [0], color=SBP_COLOR, linewidth=2, marker='o',
                   markersize=7, label='Systolic (SBP)'),
        plt.Line2D([0], [0], color=DBP_COLOR, linewidth=2, marker='s',
                   markersize=7, label='Diastolic (DBP)'),
        plt.Line2D([0], [0], marker='*', color='#F9A825', markersize=11,
                   linestyle='none', label='* Physician reviewed'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9,
              fontsize=9, edgecolor='#E0E0E0')

    fig.text(0.5, 0.07,
             'Per protocol, home blood pressure readings are collected during weeks 5-6.\n'
             'Values are reviewed by the study physician (Dr. Sarav) prior to Visit 8.',
             ha='center', fontsize=8.5, color='#616161', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_sleep_activity(pdf, sleep_weekly, steps_weekly=None, name='', study_period=''):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig,
                f'4 · Sleep & Physical Activity -- {name or "Participant"}',
                study_period)
    gs = fig.add_gridspec(1, 2, wspace=0.38, top=0.82, bottom=0.12,
                          left=0.09, right=0.93)

    ax1 = fig.add_subplot(gs[0])
    if sleep_weekly:
        x = np.arange(1, N_WEEKS + 1)
        vals = [sleep_weekly.get(w, np.nan) for w in range(1, N_WEEKS + 1)]
        colors = [SLEEP_COLOR if not np.isnan(v) else '#E0E0E0' for v in vals]
        bars = ax1.bar(x, [v if not np.isnan(v) else 0 for v in vals],
                       color=colors, width=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.05, f'{v:.1f}',
                         ha='center', va='bottom', fontsize=9,
                         fontweight='bold', color=SLEEP_COLOR)
        ax1.axhline(7, color='#BDBDBD', linewidth=1.2, linestyle='--', zorder=2)
        ax1.text(N_WEEKS + 0.1, 7.05, 'Rec. (7 hrs)',
                 fontsize=7.5, color='#9E9E9E', va='bottom')
        ax1.set_xticks(x)
        ax1.set_xticklabels(WEEK_LABELS, fontsize=10)
        ax1.set_ylabel('Hours per night', fontsize=10)
        ax1.set_title('Average Sleep Duration', fontsize=10, color='#424242', pad=6)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax1.set_facecolor('#FFFFFF')
        ax1.set_xlim(0.4, N_WEEKS + 0.6)
    else:
        placeholder_panel(ax1, 'Sleep Duration', 'Sleep duration by week\nwill appear here.')

    ax2 = fig.add_subplot(gs[1])
    if steps_weekly:
        x     = np.arange(1, N_WEEKS + 1)
        vals  = [steps_weekly.get(w, np.nan) for w in range(1, N_WEEKS + 1)]
        colors = [STEP_COLOR if not np.isnan(v) else '#E0E0E0' for v in vals]
        bars  = ax2.bar(x, [v if not np.isnan(v) else 0 for v in vals],
                        color=colors, width=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 50, f'{v:,.0f}',
                         ha='center', va='bottom', fontsize=9,
                         fontweight='bold', color=STEP_COLOR)
        ax2.set_xticks(x)
        ax2.set_xticklabels(WEEK_LABELS, fontsize=10)
        ax2.set_ylabel('Avg. steps per day', fontsize=10)
        ax2.set_title('Daily Step Count', fontsize=10, color='#424242', pad=6)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax2.set_facecolor('#FFFFFF')
        ax2.set_xlim(0.4, N_WEEKS + 0.6)
    else:
        placeholder_panel(ax2, 'Physical Activity (Steps)',
                          'Step count data will appear here\n'
                          'once CentrePoint exports are available.')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_hr_hrv(pdf, hr_weekly, hrv_weekly, name, study_period):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig,
                f'5 · Heart Rate & HRV -- {name or "Participant"}',
                f'{study_period}  ·  Nightly averages from LEAP wristband')

    ax = fig.add_axes([0.12, 0.18, 0.72, 0.62])
    ax2 = ax.twinx()

    x = np.arange(1, N_WEEKS + 1)
    hr_vals  = [hr_weekly.get(w, np.nan)  for w in range(1, N_WEEKS + 1)]
    hrv_vals = [hrv_weekly.get(w, np.nan) for w in range(1, N_WEEKS + 1)]

    valid_hr_x  = [w for w, v in zip(x, hr_vals)  if not np.isnan(v)]
    valid_hr_v  = [v for v in hr_vals  if not np.isnan(v)]
    valid_hrv_x = [w for w, v in zip(x, hrv_vals) if not np.isnan(v)]
    valid_hrv_v = [v for v in hrv_vals if not np.isnan(v)]

    if valid_hr_v:
        ax.plot(valid_hr_x, valid_hr_v, 'o-', color=HR_COLOR,
                linewidth=2.2, markersize=8, zorder=4)
        for xi, vi in zip(valid_hr_x, valid_hr_v):
            ax.text(xi, vi + 0.6, f'{vi:.0f}', ha='center', va='bottom',
                    fontsize=9, color=HR_COLOR, fontweight='bold')

    if valid_hrv_v:
        ax2.plot(valid_hrv_x, valid_hrv_v, 's--', color=RMSSD_COLOR,
                 linewidth=2.2, markersize=8, zorder=4)
        for xi, vi in zip(valid_hrv_x, valid_hrv_v):
            ax2.text(xi, vi + 0.6, f'{vi:.0f}', ha='center', va='bottom',
                     fontsize=9, color=RMSSD_COLOR, fontweight='bold')

    if not valid_hr_v and not valid_hrv_v:
        placeholder_panel(ax, '5 · Heart Rate & HRV',
                          'Nightly HR and HRV data not available for this period.\n'
                          'Run compute_hr_hrv_from_ppg.py to derive from raw PPG.')

    ax.set_xticks(range(1, N_WEEKS + 1))
    ax.set_xticklabels(WEEK_LABELS, fontsize=10)
    ax.set_ylabel('Resting Heart Rate (bpm)', fontsize=10, color=HR_COLOR)
    ax.tick_params(axis='y', labelcolor=HR_COLOR)
    ax.spines[['top']].set_visible(False)
    if valid_hrv_v:
        ax2.set_ylabel('HRV -- RMSSD (ms)', fontsize=10, color=RMSSD_COLOR)
        ax2.tick_params(axis='y', labelcolor=RMSSD_COLOR)
        ax2.spines[['top']].set_visible(False)
    else:
        ax2.set_visible(False)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0.4, N_WEEKS + 0.6)

    handles = [
        plt.Line2D([0], [0], color=HR_COLOR,    linewidth=2, marker='o',
                   markersize=7, label='Resting HR (bpm)'),
        plt.Line2D([0], [0], color=RMSSD_COLOR, linewidth=2, marker='s',
                   linestyle='--', markersize=7, label='HRV -- RMSSD (ms)'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9,
              fontsize=9, edgecolor='#E0E0E0')

    hrv_note = ('HRV (RMSSD) not shown — values derived from raw PPG peak detection are '
                'not sufficiently accurate. Will populate once a validated CentrePoint epoch export is available.')
    fig.text(0.5, 0.07,
             'Nightly resting HR computed from LEAP wristband during detected sleep windows.\n'
             'Lower resting HR and higher RMSSD are associated with greater physiological recovery.\n'
             f'{hrv_note}',
             ha='center', fontsize=8.5, color='#616161', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_timeline(pdf, sleep_weekly, bp_data, hr_weekly, hrv_weekly,
                  calm_total_weekly=None, selfdirected_weekly=None, steps_weekly=None,
                  name='', study_period=''):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig,
                f'6 · Multi-Modal Timeline Overview -- {name or "Participant"}',
                f'{study_period}  ·  Shading reflects relative level within each measure (darker = higher)')

    sbp = bp_data.get('sbp', {})

    rows = [
        ('App use (min)',       calm_total_weekly,    '#7E57C2'),
        ('Self-directed (min)', selfdirected_weekly,  '#AB47BC'),
        ('Systolic BP (mmHg)',  sbp,                  SBP_COLOR),
        ('Sleep (hrs/night)',   sleep_weekly,          SLEEP_COLOR),
        ('Steps/day',          steps_weekly,           STEP_COLOR),
        ('HRV -- RMSSD (ms)',  hrv_weekly,             RMSSD_COLOR),
    ]

    n_rows = len(rows)
    ax = fig.add_axes([0.26, 0.13, 0.66, 0.68])
    ax.set_xlim(-0.1, N_WEEKS)
    ax.set_ylim(0, n_rows)
    ax.axis('off')

    for ri, (label, data_dict, color) in enumerate(rows):
        y = n_rows - 1 - ri
        fig.text(0.255, (0.13 + (y + 0.5) * 0.68 / n_rows),
                 label, ha='right', va='center', fontsize=9, color='#424242')

        if data_dict is None:
            for ci in range(N_WEEKS):
                rect = mpatches.FancyBboxPatch(
                    (ci + 0.04, y + 0.08), 0.92, 0.84,
                    boxstyle='round,pad=0.02',
                    facecolor=PLACEHOLDER_BG, edgecolor='#E0E0E0',
                    linewidth=0.6, transform=ax.transData, clip_on=False)
                ax.add_patch(rect)
                ax.text(ci + 0.5, y + 0.5, '--', ha='center', va='center',
                        fontsize=9, color=PLACEHOLDER_TEXT)
        else:
            vals = [data_dict.get(w) for w in range(1, N_WEEKS + 1)]
            numeric = [v for v in vals if v is not None and not np.isnan(float(v))]
            vmin = min(numeric) if numeric else 0
            vmax = max(numeric) if numeric else 1
            vrange = vmax - vmin if vmax > vmin else 1

            for ci, v in enumerate(vals):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    fc   = PLACEHOLDER_BG
                    txt  = '--'
                    tc   = PLACEHOLDER_TEXT
                else:
                    alpha = 0.15 + 0.72 * (v - vmin) / vrange
                    fc    = mcolors.to_rgba(color, alpha=alpha)
                    if v >= 1000:
                        txt = f'{v / 1000:.1f}K'
                    elif v >= 10:
                        txt = f'{v:.0f}'
                    else:
                        txt = f'{v:.1f}'
                    tc    = '#212121' if alpha < 0.45 else '#FFFFFF'

                rect = mpatches.FancyBboxPatch(
                    (ci + 0.04, y + 0.08), 0.92, 0.84,
                    boxstyle='round,pad=0.02',
                    facecolor=fc, edgecolor='#E0E0E0',
                    linewidth=0.6, transform=ax.transData, clip_on=False)
                ax.add_patch(rect)
                ax.text(ci + 0.5, y + 0.5, txt, ha='center', va='center',
                        fontsize=9, color=tc, fontweight='bold')

    for ci in range(N_WEEKS):
        ax.text(ci + 0.5, n_rows + 0.15, WEEK_LABELS[ci],
                ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#424242')

    pending = []
    if calm_total_weekly is None:
        pending.append('app use')
    if steps_weekly is None:
        pending.append('step')
    pending_note = (f'{" and ".join(p.capitalize() for p in pending)} data pending integration. '
                    if pending else '')
    fig.text(0.5, 0.05,
             f'BP not collected until weeks 5-6 per protocol. {pending_note}\n'
             'Shading reflects relative level within each measure. '
             'All values are research estimates.',
             ha='center', fontsize=8, color='#9E9E9E', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_data_sources(pdf):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)

    fig.text(0.5, 0.88, 'Data Sources', ha='center', va='top',
             fontsize=16, fontweight='bold', color='#212121')

    divider = fig.add_axes([0.2, 0.855, 0.6, 0.002])
    divider.set_facecolor('#BDBDBD')
    divider.axis('off')

    bullet = '+'
    sources = [
        (f'{bullet}  Calm Health app backend dashboard',
         'App usage minutes, session timestamps, GAD-7, PHQ-8'),
        (f'{bullet}  Actigraph Leap 2 -- event marker button',
         'Self-directed mindfulness sessions; sleep onset/wake'),
        (f'{bullet}  Home blood pressure monitor',
         'Participant-owned; readings during weeks 5-6'),
        (f'{bullet}  Actigraph Leap 2 -- wristband sensors',
         'Sleep duration, steps, resting HR, HRV (derived from raw PPG)'),
    ]

    y = 0.77
    for header, detail in sources:
        fig.text(0.18, y, header, ha='left', va='top',
                 fontsize=11, fontweight='bold', color='#212121')
        fig.text(0.18, y - 0.038, detail, ha='left', va='top',
                 fontsize=10, color='#616161')
        y -= 0.115

    fig.text(0.5, 0.08,
             'These measurements are research estimates -- not a substitute for clinical evaluation.\n'
             'Blood pressure values were reviewed by the study physician prior to Visit 8.\n'
             'HR/HRV on page 5 derived from raw PPG peak detection (not Actigraph validated algorithm).',
             ha='center', fontsize=8.5, color='#9E9E9E', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_bp_trend(pdf, bp_df, dip_data, sleep_windows_raw, name, study_period, tz):
    from matplotlib.transforms import blended_transform_factory

    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig,
                f'7 · Blood Pressure Trend -- {name or "Participant"}',
                f'{study_period}  ·  Estimated from wristband  ·  Calibrated to cuff readings  ·  AHA 2017 zones shaded')

    ax = fig.add_axes([0.10, 0.38, 0.84, 0.47])

    # AHA 2017 SBP zone background shading
    ax.axhspan(50,  120, alpha=0.07, color=AHA_COLORS['Normal'],   zorder=0)
    ax.axhspan(120, 130, alpha=0.07, color=AHA_COLORS['Elevated'],  zorder=0)
    ax.axhspan(130, 140, alpha=0.07, color=AHA_COLORS['Stage 1'],   zorder=0)
    ax.axhspan(140, 200, alpha=0.07, color=AHA_COLORS['Stage 2'],   zorder=0)

    span_days = (bp_df['t'].max() - bp_df['t'].min()).days

    if span_days >= 14:
        bp_df = bp_df.copy()
        bp_df['date'] = bp_df['t'].dt.date
        daily    = bp_df.groupby('date')[['sbp', 'dbp']].mean().reset_index()
        daily['t'] = pd.to_datetime(daily['date'])
        t_plot   = daily['t']
        sbp_plot = daily['sbp']
        dbp_plot = daily['dbp']
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    elif span_days >= 2:
        bp_df = bp_df.copy()
        bp_df['date'] = bp_df['t'].dt.date
        daily    = bp_df.groupby('date')[['sbp', 'dbp']].mean().reset_index()
        daily['t'] = pd.to_datetime(daily['date'])
        t_plot   = daily['t']
        sbp_plot = daily['sbp']
        dbp_plot = daily['dbp']
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        bp_df = bp_df.sort_values('t')
        t_plot   = bp_df['t']
        sbp_plot = bp_df['sbp'].rolling(5, center=True, min_periods=1).mean()
        dbp_plot = bp_df['dbp'].rolling(5, center=True, min_periods=1).mean()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%I %p\n%b %d'))

    ax.plot(t_plot, sbp_plot, '-', color=SBP_COLOR, linewidth=1.8, zorder=4,
            label='Systolic (upper number)')
    ax.plot(t_plot, dbp_plot, '-', color=DBP_COLOR, linewidth=1.8, zorder=4,
            label='Diastolic (lower number)')

    all_v = pd.concat([pd.Series(sbp_plot).dropna(), pd.Series(dbp_plot).dropna()])
    ymin  = max(50,  float(all_v.min()) - 20)
    ymax  = min(200, float(all_v.max()) + 20)
    ax.set_ylim(ymin, ymax)

    mixed = blended_transform_factory(ax.transAxes, ax.transData)
    # AHA SBP threshold reference lines
    for thresh, label in [(120, 'Normal/Elevated (120)'), (130, 'Elevated/Stage 1 (130)'),
                           (140, 'Stage 1/Stage 2 (140)')]:
        ax.axhline(thresh, color='#BDBDBD', linewidth=1.0, linestyle='--', zorder=2)
        ax.text(1.0, thresh + 0.5, label, transform=mixed,
                fontsize=7, color='#9E9E9E', va='bottom', ha='right')
    ax.axhline(80, color='#BDBDBD', linewidth=0.8, linestyle=':', zorder=2)
    ax.text(1.0, 80.5, 'DBP threshold (80)', transform=mixed,
            fontsize=7, color='#9E9E9E', va='bottom', ha='right')

    ax.set_ylabel('Blood Pressure (mmHg)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_facecolor('#FFFFFF')
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=9)

    # AHA zone legend patches
    aha_handles = [
        mpatches.Patch(color=AHA_COLORS['Normal'],   alpha=0.4, label='Normal (<120/<80)'),
        mpatches.Patch(color=AHA_COLORS['Elevated'],  alpha=0.4, label='Elevated (120-129/<80)'),
        mpatches.Patch(color=AHA_COLORS['Stage 1'],   alpha=0.4, label='Stage 1 (130-139 or 80-89)'),
        mpatches.Patch(color=AHA_COLORS['Stage 2'],   alpha=0.4, label='Stage 2 (>=140 or >=90)'),
    ]
    line_handles = [
        plt.Line2D([0], [0], color=SBP_COLOR, linewidth=2, label='Systolic (upper number)'),
        plt.Line2D([0], [0], color=DBP_COLOR, linewidth=2, label='Diastolic (lower number)'),
    ]
    ax.legend(handles=line_handles + aha_handles, loc='upper right', framealpha=0.9,
              fontsize=8, edgecolor='#E0E0E0', ncol=2)

    # Nocturnal dip stats box
    if dip_data:
        nights_list = dip_data.get('nights', [])
        summary     = dip_data.get('summary', {})
        mean_wake   = np.mean([n['mean_wake_sbp']  for n in nights_list]) if nights_list else None
        mean_sleep  = np.mean([n['mean_sleep_sbp'] for n in nights_list]) if nights_list else None
        dip_pct     = summary.get('mean_dip_pct', 0)
        threshold   = dip_data.get('dipper_threshold_pct', 10.0)
        dip_status  = ('Healthy dipper' if dip_pct >= threshold
                       else f'Below healthy threshold (goal: >=10%)')
        lines = []
        if mean_wake  is not None:
            lines.append(f'Average daytime SBP:   {mean_wake:.0f} mmHg')
        if mean_sleep is not None:
            lines.append(f'Average nighttime SBP: {mean_sleep:.0f} mmHg')
        lines.append(f'Nocturnal dip:         {dip_pct:.1f}%  --  {dip_status}')
        lines.append('')
        lines.append('Estimates from wristband -- bring to care team for interpretation.')
        fig.text(0.12, 0.33, '\n'.join(lines),
                 ha='left', va='top', fontsize=8.5, color='#424242',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                           edgecolor='#E0E0E0', linewidth=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_weekly_snapshot(pdf, week_num, study_start, tz,
                          sleep_daily, steps_daily, hr_daily, hrv_daily,
                          bp_stage_daily, calm_daily, bp_df):
    """Informal daily data table + mini BP trend for one study week."""
    week_start = pd.Timestamp(study_start) + pd.Timedelta(weeks=week_num - 1)
    week_end   = week_start + pd.Timedelta(days=6)
    dates = [(week_start + pd.Timedelta(days=i)).date().isoformat() for i in range(7)]
    today = pd.Timestamp.today().date().isoformat()

    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig,
                f'Week {week_num} -- Daily Snapshot  (informal)',
                f'{week_start.strftime("%b %d")} - {week_end.strftime("%b %d, %Y")}')

    # ── Table ─────────────────────────────────────────────────────────────────
    col_headers = ['Date', 'Day', 'Sleep\n(hrs)', 'Steps', 'Resting\nHR (bpm)',
                   'RMSSD\n(ms)', 'AHA Stage\n(SBP/DBP est.)', 'Calm\nSessions']
    col_widths  = [0.120, 0.060, 0.085, 0.090, 0.095, 0.080, 0.195, 0.085]
    col_x = [sum(col_widths[:i]) for i in range(len(col_widths))]
    # Normalize to [0,1] range within axes
    total_w = sum(col_widths)
    col_x_n = [x / total_w for x in col_x]
    col_w_n = [w / total_w for w in col_widths]

    ax_tbl = fig.add_axes([0.04, 0.44, 0.92, 0.47])
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)
    ax_tbl.axis('off')

    n_rows = len(dates)
    hdr_h  = 1.0 / (n_rows + 1.2)
    row_h  = (1.0 - hdr_h) / n_rows

    # Header
    hdr_y = 1.0 - hdr_h * 0.5
    for cx, cw, lbl in zip(col_x_n, col_w_n, col_headers):
        ax_tbl.text(cx + cw / 2, hdr_y, lbl, ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color='#424242',
                    transform=ax_tbl.transAxes)
    # Header underline
    ax_tbl.plot([0, 1], [1.0 - hdr_h, 1.0 - hdr_h], color='#BDBDBD',
                linewidth=0.8, transform=ax_tbl.transAxes, clip_on=False)

    for ri, date in enumerate(dates):
        row_y  = (1.0 - hdr_h) - (ri + 0.5) * row_h
        is_future = date > today

        # Alternating row background
        bg_color = '#F5F5F5' if ri % 2 == 0 else '#FFFFFF'
        rect = mpatches.FancyBboxPatch(
            (0, (1.0 - hdr_h) - (ri + 1) * row_h),
            1.0, row_h,
            boxstyle='square,pad=0',
            facecolor=bg_color, edgecolor='none',
            transform=ax_tbl.transAxes, clip_on=False, zorder=0)
        ax_tbl.add_patch(rect)

        day_name  = pd.Timestamp(date).strftime('%a')
        if is_future:
            row_vals = [date, day_name, '--', '--', '--', '--', '--', '--']
            row_cols = [PLACEHOLDER_TEXT] * 8
            row_bold = [False] * 8
        else:
            sleep_v = sleep_daily.get(date)
            steps_v = steps_daily.get(date)
            hr_v    = hr_daily.get(date)
            hrv_v   = hrv_daily.get(date)
            bp_info = bp_stage_daily.get(date)
            calm_v  = calm_daily.get(date, 0)

            sleep_s = f'{sleep_v:.1f}' if sleep_v is not None else '--'
            steps_s = f'{int(steps_v):,}' if steps_v is not None else '--'
            hr_s    = f'{hr_v:.0f}'    if hr_v    is not None else '--'
            hrv_s   = f'{hrv_v:.0f}'  if hrv_v   is not None else '--'
            calm_s  = str(int(calm_v)) if calm_v else '--'

            if bp_info:
                stage, bp_color, sbp, dbp = bp_info
                bp_s   = f'{stage}\n{sbp:.0f}/{dbp:.0f}'
                bp_col = bp_color
            else:
                bp_s, bp_col = '--', PLACEHOLDER_TEXT

            row_vals = [date, day_name, sleep_s, steps_s, hr_s, hrv_s, bp_s, calm_s]
            row_cols = ['#424242', '#616161', SLEEP_COLOR if sleep_v else PLACEHOLDER_TEXT,
                        STEP_COLOR if steps_v else PLACEHOLDER_TEXT,
                        HR_COLOR if hr_v else PLACEHOLDER_TEXT,
                        RMSSD_COLOR if hrv_v else PLACEHOLDER_TEXT,
                        bp_col,
                        '#7E57C2' if calm_v else PLACEHOLDER_TEXT]
            row_bold = [False, False,
                        sleep_v is not None, steps_v is not None,
                        hr_v is not None, hrv_v is not None,
                        bp_info is not None, calm_v > 0]

        for cx, cw, val, col, bold in zip(col_x_n, col_w_n, row_vals, row_cols, row_bold):
            ax_tbl.text(cx + cw / 2, row_y, val,
                        ha='center', va='center', fontsize=8,
                        color=col, fontweight='bold' if bold else 'normal',
                        transform=ax_tbl.transAxes)

    # ── Mini BP chart ─────────────────────────────────────────────────────────
    ax_bp = fig.add_axes([0.08, 0.07, 0.88, 0.30])

    has_bp = False
    if bp_df is not None and not bp_df.empty:
        week_start_ts = pd.Timestamp(week_start).tz_localize(tz)
        week_end_ts   = week_start_ts + pd.Timedelta(days=7)
        bp_week = bp_df[(bp_df['t'] >= week_start_ts) & (bp_df['t'] < week_end_ts)].copy()

        if not bp_week.empty:
            has_bp = True
            bp_week['date_col'] = bp_week['t'].dt.date
            daily_bp = bp_week.groupby('date_col')[['sbp', 'dbp']].mean().reset_index()
            daily_bp['t_dt'] = pd.to_datetime(daily_bp['date_col'])

            # AHA zone bands
            ax_bp.axhspan(50,  120, alpha=0.08, color=AHA_COLORS['Normal'],   zorder=0)
            ax_bp.axhspan(120, 130, alpha=0.08, color=AHA_COLORS['Elevated'],  zorder=0)
            ax_bp.axhspan(130, 140, alpha=0.08, color=AHA_COLORS['Stage 1'],   zorder=0)
            ax_bp.axhspan(140, 200, alpha=0.08, color=AHA_COLORS['Stage 2'],   zorder=0)

            ax_bp.plot(daily_bp['t_dt'], daily_bp['sbp'], 'o-', color=SBP_COLOR,
                       linewidth=2, markersize=6, label='SBP (est.)', zorder=4)
            ax_bp.plot(daily_bp['t_dt'], daily_bp['dbp'], 's-', color=DBP_COLOR,
                       linewidth=2, markersize=6, label='DBP (est.)', zorder=4)

            for _, row in daily_bp.iterrows():
                ax_bp.text(row['t_dt'], row['sbp'] + 1.0, f'{row["sbp"]:.0f}',
                           ha='center', va='bottom', fontsize=7.5, color=SBP_COLOR)
                ax_bp.text(row['t_dt'], row['dbp'] - 1.5, f'{row["dbp"]:.0f}',
                           ha='center', va='top', fontsize=7.5, color=DBP_COLOR)

            for thresh in [120, 130, 140]:
                ax_bp.axhline(thresh, color='#BDBDBD', linewidth=0.7,
                              linestyle='--', zorder=2)

            ax_bp.xaxis.set_major_locator(mdates.DayLocator())
            ax_bp.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax_bp.set_ylabel('BP (mmHg)', fontsize=9)
            ax_bp.set_title('Continuous BP Estimate -- daily median  (research estimate, not clinical)',
                            fontsize=8, color='#616161', pad=3)
            ax_bp.spines[['top', 'right']].set_visible(False)
            ax_bp.grid(axis='y', color=GRID_COLOR, linewidth=0.6, zorder=0)
            ax_bp.set_facecolor('#FFFFFF')
            ax_bp.legend(fontsize=8, loc='upper right', framealpha=0.85, edgecolor='#E0E0E0')

            all_v = pd.concat([daily_bp['sbp'], daily_bp['dbp']]).dropna()
            if not all_v.empty:
                ax_bp.set_ylim(max(50, float(all_v.min()) - 12),
                               min(200, float(all_v.max()) + 12))

    if not has_bp:
        placeholder_panel(ax_bp, 'No wristband BP data this week',
                          'Wristband estimates available for weeks 4-5 (May 21-30).')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_cuff_vs_lstm(pdf, cuff_csv, p3_pred_csv, p5_pred_csv, name, tz):
    """Informal page: cuff ground-truth vs Phase 3 and Phase 5 LSTM predictions."""
    from scipy.stats import pearsonr
    from matplotlib.transforms import blended_transform_factory

    # ── Load data ─────────────────────────────────────────────────────────────
    cuff = pd.read_csv(cuff_csv)
    cuff['t'] = pd.to_datetime(cuff['timestamp_utc'], utc=True).dt.tz_convert(tz)
    cuff = cuff.sort_values('t').reset_index(drop=True)

    p3 = pd.read_csv(p3_pred_csv)
    p5 = pd.read_csv(p5_pred_csv)
    p3['t'] = pd.to_datetime(p3['window_start_utc'], utc=True).dt.tz_convert(tz)
    p5['t'] = pd.to_datetime(p5['window_start_utc'], utc=True).dt.tz_convert(tz)

    # ── Match cuff readings to nearest prediction window (<=15 min) ───────────
    def match_nearest(cuff_df, pred_df, max_gap_min=15):
        pairs = []
        for _, row in cuff_df.iterrows():
            delta = (pred_df['t'] - row['t']).abs()
            idx = delta.idxmin()
            if delta[idx].total_seconds() <= max_gap_min * 60:
                pairs.append({
                    't':        row['t'],
                    'cuff_sbp': float(row['sbp_mmhg']),
                    'cuff_dbp': float(row['dbp_mmhg']),
                    'pred_sbp': float(pred_df.loc[idx, 'sbp_mmhg']),
                    'pred_dbp': float(pred_df.loc[idx, 'dbp_mmhg']),
                    'gap_min':  delta[idx].total_seconds() / 60,
                })
        return pd.DataFrame(pairs)

    m3 = match_nearest(cuff, p3)
    m5 = match_nearest(cuff, p5)

    # Re-calibrate predictions to the matched cuff mean (removes systematic offset)
    for m in [m3, m5]:
        if m.empty:
            continue
        bias = m['pred_sbp'].mean() - m['cuff_sbp'].mean()
        m['pred_sbp_cal'] = m['pred_sbp'] - bias
        bias_d = m['pred_dbp'].mean() - m['cuff_dbp'].mean()
        m['pred_dbp_cal'] = m['pred_dbp'] - bias_d

    def stats(m):
        if len(m) < 3:
            return dict(r=np.nan, p=np.nan, mae=np.nan, rmse=np.nan, bias=np.nan, n=len(m))
        r, p = pearsonr(m['cuff_sbp'], m['pred_sbp'])
        resid = m['pred_sbp_cal'] - m['cuff_sbp']
        return dict(r=r, p=p, mae=float(resid.abs().mean()), rmse=float(np.sqrt((resid**2).mean())),
                    bias=float(m['pred_sbp'].mean() - m['cuff_sbp'].mean()), n=len(m))

    s3, s5 = stats(m3), stats(m5)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    page_header(fig, f'LSTM Model Validation -- {name or "SUBJ01"}  (informal)',
                'Phase 5 (Aurora-BP trained) vs Phase 3 (ICU-trained)  ·  n=12 matched cuff readings')

    # ── 1. Time-series ────────────────────────────────────────────────────────
    ax_ts = fig.add_axes([0.08, 0.47, 0.86, 0.41])

    # AHA zone bands
    ax_ts.axhspan(50,  120, alpha=0.07, color=AHA_COLORS['Normal'],   zorder=0)
    ax_ts.axhspan(120, 130, alpha=0.07, color=AHA_COLORS['Elevated'],  zorder=0)
    ax_ts.axhspan(130, 140, alpha=0.07, color=AHA_COLORS['Stage 1'],   zorder=0)
    ax_ts.axhspan(140, 200, alpha=0.07, color=AHA_COLORS['Stage 2'],   zorder=0)

    # AHA threshold lines
    for thresh in [120, 130, 140]:
        ax_ts.axhline(thresh, color='#BDBDBD', linewidth=0.8, linestyle='--', zorder=2)

    # Phase 5 prediction line (daily mean, re-centered)
    p5_bias = m5['pred_sbp'].mean() - m5['cuff_sbp'].mean() if not m5.empty else 0
    p5_daily = p5.copy()
    p5_daily['sbp_cal'] = p5_daily['sbp_mmhg'] - p5_bias
    p5_daily['date'] = p5_daily['t'].dt.date
    p5_agg = p5_daily.groupby('date')['sbp_cal'].mean().reset_index()
    p5_agg['t_dt'] = pd.to_datetime(p5_agg['date'])

    p3_bias = m3['pred_sbp'].mean() - m3['cuff_sbp'].mean() if not m3.empty else 0
    p3_daily = p3.copy()
    p3_daily['sbp_cal'] = p3_daily['sbp_mmhg'] - p3_bias
    p3_daily['date'] = p3_daily['t'].dt.date
    p3_agg = p3_daily.groupby('date')['sbp_cal'].mean().reset_index()
    p3_agg['t_dt'] = pd.to_datetime(p3_agg['date'])

    ax_ts.plot(p3_agg['t_dt'], p3_agg['sbp_cal'], '-', color='#9E9E9E', linewidth=1.6,
               zorder=3, label=f'Phase 3 SBP est. (daily avg, r={s3["r"]:.2f})', alpha=0.8)
    ax_ts.plot(p5_agg['t_dt'], p5_agg['sbp_cal'], '-', color='#1565C0', linewidth=2.0,
               zorder=4, label=f'Phase 5 SBP est. (daily avg, r={s5["r"]:.2f})')

    # Cuff readings — all 20 (matched ones filled, unmatched open)
    matched_times = set(m5['t'].dt.date.astype(str).tolist()) if not m5.empty else set()
    for _, row in cuff.iterrows():
        min_gap = (p5['t'] - row['t']).abs().dt.total_seconds().min()
        in_window = (not m5.empty) and (min_gap <= 900)
        marker = 'o' if in_window else 'x'
        ec = SBP_COLOR if in_window else 'none'
        ax_ts.scatter(row['t'].to_pydatetime(), row['sbp_mmhg'],
                      s=70, color=SBP_COLOR, zorder=6,
                      marker=marker, edgecolors=ec, linewidths=1.5,
                      alpha=1.0 if in_window else 0.5)
        ax_ts.text(row['t'].to_pydatetime(), row['sbp_mmhg'] + 1.5,
                   f'{int(row["sbp_mmhg"])}', ha='center', va='bottom',
                   fontsize=6.5, color=SBP_COLOR, zorder=7)

    # Error bars: vertical lines from Phase5 match to cuff
    for _, row in m5.iterrows():
        ax_ts.plot([row['t'].to_pydatetime(), row['t'].to_pydatetime()],
                   [row['pred_sbp_cal'], row['cuff_sbp']],
                   color='#1565C0', linewidth=0.7, alpha=0.4, zorder=3)

    all_y = list(cuff['sbp_mmhg']) + list(p5['sbp_mmhg'] - p5_bias) + list(p3['sbp_mmhg'] - p3_bias)
    ax_ts.set_ylim(max(50, min(all_y) - 10), min(200, max(all_y) + 12))
    ax_ts.set_ylabel('SBP (mmHg)', fontsize=9)
    ax_ts.xaxis.set_major_locator(mdates.DayLocator())
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax_ts.xaxis.get_majorticklabels(), fontsize=8.5, rotation=20, ha='right')
    ax_ts.spines[['top', 'right']].set_visible(False)
    ax_ts.grid(axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax_ts.set_facecolor('#FFFFFF')

    handles_ts = [
        plt.Line2D([0], [0], color='#1565C0', linewidth=2, label=f'Phase 5 est. (r={s5["r"]:.2f}, n={s5["n"]})'),
        plt.Line2D([0], [0], color='#9E9E9E', linewidth=1.6, label=f'Phase 3 est. (r={s3["r"]:.2f}, n={s3["n"]})'),
        plt.Line2D([0], [0], marker='o', color=SBP_COLOR, linestyle='none',
                   markersize=6, label='Cuff SBP (matched)'),
        plt.Line2D([0], [0], marker='x', color=SBP_COLOR, linestyle='none',
                   markersize=6, label='Cuff SBP (outside window)'),
    ]
    ax_ts.legend(handles=handles_ts, fontsize=8, loc='upper right',
                 framealpha=0.9, edgecolor='#E0E0E0')

    # ── 2. Scatter plots ─────────────────────────────────────────────────────
    for col_offset, m, s, label, color in [
        (0.07,  m3, s3, 'Phase 3', '#9E9E9E'),
        (0.56,  m5, s5, 'Phase 5', '#1565C0'),
    ]:
        ax_sc = fig.add_axes([col_offset, 0.08, 0.37, 0.31])

        if m.empty:
            placeholder_panel(ax_sc, label, 'No matched pairs')
            continue

        ax_sc.axhspan(50,  120, alpha=0.07, color=AHA_COLORS['Normal'],  zorder=0)
        ax_sc.axhspan(120, 130, alpha=0.07, color=AHA_COLORS['Elevated'], zorder=0)
        ax_sc.axhspan(130, 140, alpha=0.07, color=AHA_COLORS['Stage 1'],  zorder=0)
        ax_sc.axhspan(140, 200, alpha=0.07, color=AHA_COLORS['Stage 2'],  zorder=0)

        vmin = min(m['cuff_sbp'].min(), m['pred_sbp_cal'].min()) - 5
        vmax = max(m['cuff_sbp'].max(), m['pred_sbp_cal'].max()) + 5
        ax_sc.plot([vmin, vmax], [vmin, vmax], '--', color='#BDBDBD',
                   linewidth=1.0, zorder=2, label='Perfect prediction')

        ax_sc.scatter(m['cuff_sbp'], m['pred_sbp_cal'],
                      s=55, color=color, edgecolors='white',
                      linewidths=0.6, zorder=5, alpha=0.9)

        # Label each point with date
        for _, row in m.iterrows():
            ax_sc.text(row['cuff_sbp'] + 0.5, row['pred_sbp_cal'],
                       row['t'].strftime('%m/%d'), fontsize=5.5,
                       color='#616161', va='center')

        ax_sc.set_xlim(vmin, vmax)
        ax_sc.set_ylim(vmin, vmax)
        ax_sc.set_xlabel('Cuff SBP (mmHg)', fontsize=8.5)
        ax_sc.set_ylabel('Predicted SBP (mmHg)', fontsize=8.5)
        ax_sc.set_facecolor('#FFFFFF')
        ax_sc.spines[['top', 'right']].set_visible(False)
        ax_sc.grid(color=GRID_COLOR, linewidth=0.7, zorder=0)

        p_str = f'p={s["p"]:.3f}' if not np.isnan(s['p']) else ''
        stats_txt = (f'{label}\n'
                     f'r = {s["r"]:.3f}  {p_str}\n'
                     f'MAE = {s["mae"]:.1f} mmHg (bias-removed)\n'
                     f'Original bias = {s["bias"]:+.1f} mmHg\n'
                     f'n = {s["n"]} matched readings')
        ax_sc.text(0.04, 0.97, stats_txt, transform=ax_sc.transAxes,
                   fontsize=7.5, va='top', color='#212121',
                   bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                             edgecolor='#E0E0E0', linewidth=0.7))

        # AHA categorical accuracy
        n_exact = n_within1 = 0
        for _, row in m.iterrows():
            cuff_stage, _ = classify_aha_stage(row['cuff_sbp'],    row['cuff_dbp'])
            pred_stage, _ = classify_aha_stage(row['pred_sbp_cal'], row['pred_dbp_cal'])
            stage_order = ['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis']
            ci = stage_order.index(cuff_stage) if cuff_stage in stage_order else 2
            pi = stage_order.index(pred_stage) if pred_stage in stage_order else 2
            if ci == pi:
                n_exact += 1
            if abs(ci - pi) <= 1:
                n_within1 += 1

        n_tot = len(m)
        aha_txt = (f'AHA stage exact: {n_exact}/{n_tot} ({100*n_exact/n_tot:.0f}%)\n'
                   f'AHA within 1 stage: {n_within1}/{n_tot} ({100*n_within1/n_tot:.0f}%)')
        ax_sc.text(0.04, 0.52, aha_txt, transform=ax_sc.transAxes,
                   fontsize=7, va='top', color='#424242',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#EDE7F6',
                             edgecolor='#D1C4E9', linewidth=0.7))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    tz       = args.timezone
    subj     = args.subject_id
    name     = args.name or subj
    inf_dir  = args.inference_dir
    data_dir = args.data_dir
    out_dir  = Path(args.out_dir or inf_dir) / subj
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'patient_report_full.pdf'

    print('Loading data ...')
    if args.leap_dir:
        sleep_windows_raw = load_sleep_windows_from_leap(args.leap_dir, subj, tz)
        print(f'LEAP sleep windows loaded: {len(sleep_windows_raw)} nights')
    else:
        sleep_windows_raw = load_sleep_windows_raw(inf_dir, subj)

    # Study start
    if args.study_start:
        study_start = args.study_start
    elif sleep_windows_raw:
        study_start = sleep_windows_raw[0]['date']
    else:
        bp_path = (Path(inf_dir) / subj / 'bp_trend_calibrated.csv')
        if not bp_path.exists():
            bp_path = Path(inf_dir) / subj / 'bp_trend.csv'
        bp_tmp = pd.read_csv(bp_path)
        study_start = (pd.to_datetime(bp_tmp['datetime_utc'].iloc[0], utc=True)
                       .tz_convert(tz).date().isoformat())

    study_end_projected = pd.Timestamp(study_start) + timedelta(weeks=N_WEEKS)
    today = pd.Timestamp.today().normalize()
    study_end_ts = min(study_end_projected, today)
    current_week = min(N_WEEKS, int((today - pd.Timestamp(study_start)).days // 7 + 1))
    study_period = (
        f"Weeks 1-{current_week} of {N_WEEKS}  "
        f"({pd.Timestamp(study_start).strftime('%m/%d/%y')} - "
        f"{study_end_ts.strftime('%m/%d/%y')})"
    )
    report_date = pd.Timestamp.now().strftime('%B %d, %Y')

    # Weekly aggregates
    sleep_weekly   = load_sleep_weekly(sleep_windows_raw, study_start)
    steps_weekly   = load_steps_weekly(args.leap_dir, subj, study_start) if args.leap_dir else {}
    bp_data, reviewed_weeks = load_cuff_bp_weekly(args.cuff_csv, subj, study_start, tz)

    # HR/HRV (weekly + daily)
    hr_weekly, hrv_weekly, hr_daily, hrv_daily = load_hr_hrv_weekly(
        data_dir, sleep_windows_raw, study_start, tz)
    # HRV derived from raw PPG peak detection — not accurate enough; suppress until
    # a validated CentrePoint epoch export (HR/HRV channels) is available.
    hrv_weekly = {}
    hrv_daily  = {}

    # BP trend (continuous)
    bp_trend_df, dip_data = load_bp_trend_data(inf_dir, subj, tz)

    if steps_weekly:
        print(f'Steps loaded: {steps_weekly}')

    # Calm Health (weekly + daily)
    calm_mindfulness = None
    calm_other       = None
    calm_count_mode  = False
    calm_daily       = {}
    if args.calm_data_dir and args.calm_member_id:
        calm_study_start = args.calm_study_start or study_start
        calm_mindfulness, calm_other = load_calm_weekly(
            args.calm_data_dir, args.calm_member_id, calm_study_start)
        calm_daily = load_calm_daily(args.calm_data_dir, args.calm_member_id)
        calm_count_mode = True
        print(f'Calm sessions loaded: mindfulness={sum(calm_mindfulness.values())} '
              f'other={sum(calm_other.values())}')

    # Daily data for snapshot pages
    sleep_daily    = {n['date']: n['duration_hours'] for n in sleep_windows_raw}
    steps_daily    = load_steps_daily(args.leap_dir, subj) if args.leap_dir else {}
    bp_stage_daily = compute_daily_bp_stage(bp_trend_df, tz)

    print(f'Generating PDF: {out_path}')
    with PdfPages(out_path) as pdf:
        # ── Official pages ───────────────────────────────────────────────────
        page_cover(pdf, name, subj, study_period, report_date, bp_reviewer='Dr. Sarav')

        page_calm_health(pdf, name,
                         calm_mindfulness=calm_mindfulness,
                         calm_other=calm_other,
                         count_mode=calm_count_mode)

        page_home_bp(pdf, bp_data, reviewed_weeks, name, study_period)

        page_sleep_activity(pdf, sleep_weekly, steps_weekly=steps_weekly or None,
                            name=name, study_period=study_period)

        page_hr_hrv(pdf, hr_weekly, hrv_weekly, name, study_period)

        calm_total_weekly = None
        if calm_mindfulness:
            calm_total_weekly = {w: calm_mindfulness.get(w, 0) + (calm_other.get(w, 0) if calm_other else 0)
                                 for w in range(1, N_WEEKS + 1)}

        page_timeline(pdf, sleep_weekly, bp_data, hr_weekly, hrv_weekly,
                      calm_total_weekly=calm_total_weekly,
                      steps_weekly=steps_weekly or None,
                      name=name, study_period=study_period)

        if bp_trend_df is not None:
            page_bp_trend(pdf, bp_trend_df, dip_data, sleep_windows_raw,
                          name, study_period, tz)

        page_data_sources(pdf)

        # ── Informal daily snapshot pages (one per study week) ───────────────
        for wk in range(1, N_WEEKS + 1):
            page_weekly_snapshot(
                pdf, wk, study_start, tz,
                sleep_daily=sleep_daily,
                steps_daily=steps_daily,
                hr_daily=hr_daily,
                hrv_daily=hrv_daily,
                bp_stage_daily=bp_stage_daily,
                calm_daily=calm_daily,
                bp_df=bp_trend_df,
            )

        # ── LSTM validation page (if prediction CSVs provided) ──────────────
        p5_csv = args.p5_pred_csv
        p3_csv = args.p3_pred_csv
        # Default paths for SUBJ01
        if p5_csv is None:
            # Prefer 10-day Phase 5 run (full PPG set, all 20 cuff readings in window)
            default_p5_10day = Path(r'D:\Mindful Kidney\Biometric Report\data\centrepoint_mktest01_phase5_10day\predictions.csv')
            default_p5_8day  = Path(r'D:\Mindful Kidney\LSTM-model-phase 5\mk-bp-pipeline\benchmark\results\centrepoint_mktest01_8day_phase5\predictions.csv')
            if default_p5_10day.exists():
                p5_csv = str(default_p5_10day)
                print('Using Phase 5 10-day predictions for validation page.')
            elif default_p5_8day.exists():
                p5_csv = str(default_p5_8day)
                print('Using Phase 5 8-day predictions for validation page (10-day not found).')
        if p3_csv is None:
            default_p3 = Path(r'D:\Mindful Kidney\LSTM-model-phase 5\mk-bp-pipeline\benchmark\results\centrepoint_mktest01_8day_phase3\predictions.csv')
            if default_p3.exists():
                p3_csv = str(default_p3)
        if args.cuff_csv and p5_csv and p3_csv:
            print('Generating LSTM validation page ...')
            page_cuff_vs_lstm(pdf, args.cuff_csv, p3_csv, p5_csv, name, tz)

        meta = pdf.infodict()
        meta['Title']   = f'Biometric Report -- {name}'
        meta['Subject'] = 'Mindful Kidney Study -- Biometric Report'

    print(f'Done. Saved to: {out_path}')


if __name__ == '__main__':
    main()

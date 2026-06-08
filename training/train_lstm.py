#!/usr/bin/env python3
"""
Train LSTM model for within-person relative BP estimation from PPG features.
Architecture: Radha et al. 2019 (Physiol. Meas. 40 025006)
  Input  ->  Dense(8, relu)  ->  LSTM(32, return_sequences)  ->  Dense(3)
  Predicts relative SBP, DBP, MAP at every timestep (sequence-to-sequence).

Loss: MSE amplified by |SBP - train_mean_SBP| + 1 (Radha et al. Sec. 3.3.4)
  -- upweights deviant BP values so the model learns nocturnal dip patterns

Input:  .npz files from mimic_preprocessed/
        features (W, 13), sbp/dbp/map_bp (W,) relative mmHg per record
Output: lstm_model/
  best_model.keras         best checkpoint by val loss
  feat_mean.npy / feat_std.npy   normalization stats (needed at inference)
  training_log.csv         epoch-by-epoch loss
  training_history.json    full history dict
  eval_results.json        test RMSE + Pearson r for SBP, DBP, MAP

Usage (local):
  python train_lstm.py

Usage (YCRC SLURM):
  sbatch slurm_train.sh     (see slurm_train.sh in same directory)

Requirements: pip install tensorflow numpy scipy
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(r'mimic_preprocessed')
OUT_DIR  = Path(r'lstm_model')

SEQ_LEN       = 20      # windows per sequence  (20 x 15 s step = 5 min context)
SEQ_STEP_TRAIN = 10     # stride between train sequences (50% overlap)
SEQ_STEP_EVAL  = 1      # stride at eval for maximum coverage

TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.10
RANDOM_SEED = 42

DENSE_UNITS = 8         # first hidden layer  (Radha et al.)
LSTM_UNITS  = 32        # LSTM layer          (Radha et al.)
DROPOUT     = 0.3       # applied after dense and after LSTM
N_FEATURES  = 13
N_OUTPUTS   = 3         # sbp, dbp, map

EPOCHS     = 300
BATCH_SIZE = 64
PATIENCE   = 30         # early stopping (Radha: 20; increased for capped WeightedMSE)
LR         = 1e-3
MIN_WINDOWS = SEQ_LEN + 1
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description='Train Radha et al. LSTM for BP')
    p.add_argument('--data-dir',        default=str(DATA_DIR))
    p.add_argument('--out-dir',         default=str(OUT_DIR))
    p.add_argument('--epochs',          type=int,   default=EPOCHS)
    p.add_argument('--batch-size',      type=int,   default=BATCH_SIZE)
    p.add_argument('--gpus',            type=str,   default='',
                   help='CUDA_VISIBLE_DEVICES string, e.g. "0" or "0,1"')
    p.add_argument('--mixed-precision', action='store_true',
                   help='Enable float16 mixed precision (for A100/V100 GPUs)')
    p.add_argument('--seq-len',         type=int,   default=SEQ_LEN)
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(data_dir):
    """Return list of (features (W,13), labels (W,3), record_id) tuples."""
    records = []
    skipped = 0
    for path in sorted(Path(data_dir).glob('*.npz')):
        d = np.load(path, allow_pickle=True)
        feats = d['features'].astype(np.float32)
        sbp   = d['sbp'].astype(np.float32)
        dbp   = d['dbp'].astype(np.float32)
        mp    = d['map_bp'].astype(np.float32)
        if len(feats) < MIN_WINDOWS:
            skipped += 1
            continue
        valid = ~np.isnan(feats).any(axis=1)
        feats, sbp, dbp, mp = feats[valid], sbp[valid], dbp[valid], mp[valid]
        if len(feats) < MIN_WINDOWS:
            skipped += 1
            continue
        labels = np.stack([sbp, dbp, mp], axis=-1)  # (W, 3)
        records.append((feats, labels, path.stem))
    if skipped:
        print(f"  Skipped {skipped} records with < {MIN_WINDOWS} windows")
    return records


def normalize_features(records, mean=None, std=None):
    """
    Z-score each of the 13 features across all windows.
    mean/std are computed from the training set and reused for val/test.
    Returns (normalized_records, mean (13,), std (13,)).
    """
    if mean is None:
        all_feats = np.concatenate([r[0] for r in records], axis=0)
        mean = all_feats.mean(axis=0)   # (13,)
        std  = all_feats.std(axis=0) + 1e-8
    normed = [((f - mean) / std, l, name) for f, l, name in records]
    return normed, mean, std


# ── Sequence construction ─────────────────────────────────────────────────────

def make_sequences(records, seq_len, seq_step):
    """
    Slice each record into overlapping windows of length seq_len.
    Returns X (N, seq_len, 13), y (N, seq_len, 3).
    """
    X_list, y_list = [], []
    for feats, labels, _ in records:
        n = len(feats)
        for start in range(0, n - seq_len + 1, seq_step):
            X_list.append(feats[start:start + seq_len])
            y_list.append(labels[start:start + seq_len])
    if not X_list:
        return (np.empty((0, seq_len, N_FEATURES), np.float32),
                np.empty((0, seq_len, N_OUTPUTS),  np.float32))
    return (np.stack(X_list).astype(np.float32),
            np.stack(y_list).astype(np.float32))


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(seq_len):
    """
    Radha et al. 2019 three-layer LSTM network:
      TimeDistributed Dense(8, relu) -> LSTM(32) -> TimeDistributed Dense(3)
    Applied at every timestep (sequence-to-sequence mapping).
    """
    inp = tf.keras.Input(shape=(seq_len, N_FEATURES), name='ppg_features')

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(DENSE_UNITS, activation='relu'),
        name='dense_hidden'
    )(inp)
    x = tf.keras.layers.Dropout(DROPOUT, name='dropout_dense')(x)

    x = tf.keras.layers.LSTM(
        LSTM_UNITS, return_sequences=True, name='lstm'
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT, name='dropout_lstm')(x)

    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(N_OUTPUTS),
        name='bp_output'
    )(x)

    return tf.keras.Model(inputs=inp, outputs=out, name='radha_lstm')


# ── Loss ──────────────────────────────────────────────────────────────────────

class WeightedMSE(tf.keras.losses.Loss):
    """
    MSE amplified by (|SBP_true - train_mean_SBP| + 1).
    Radha et al. Sec. 3.3.4: upweights deviant BP values to improve
    sensitivity to the nocturnal dip (which is a deviant pattern by definition).
    """
    def __init__(self, train_mean_sbp=0.0, **kwargs):
        super().__init__(**kwargs)
        self.train_mean_sbp = float(train_mean_sbp)

    def call(self, y_true, y_pred):
        sbp_true = tf.cast(y_true[..., 0], tf.float32)
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.cast(y_pred, tf.float32)
        # Cap at 5x: ICU episodes reach |delta|=80 mmHg (81x) vs Radha's ambulatory
        # max of ~31x. Without the cap the gradient is dominated by vasopressor/sepsis
        # episodes unique to each patient, preventing cross-subject generalization.
        weight   = tf.minimum(tf.abs(sbp_true - self.train_mean_sbp) + 1.0, 5.0)
        mse      = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)  # (batch, seq)
        return tf.reduce_mean(mse * weight)

    def get_config(self):
        cfg = super().get_config()
        cfg['train_mean_sbp'] = self.train_mean_sbp
        return cfg


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, records, seq_len, seq_step, label='test'):
    """
    Per-record RMSE and Pearson r for SBP, DBP, MAP.
    Uses only the last-step prediction of each sequence to avoid
    counting overlapping windows multiple times.
    """
    sbp_t, sbp_p = [], []
    dbp_t, dbp_p = [], []
    map_t, map_p = [], []

    for feats, labels, rec_id in records:
        X, y = make_sequences([(feats, labels, rec_id)], seq_len, seq_step)
        if len(X) == 0:
            continue
        preds = model.predict(X, verbose=0)    # (N, seq_len, 3)
        sbp_t.extend(y[:, -1, 0].tolist())
        sbp_p.extend(preds[:, -1, 0].tolist())
        dbp_t.extend(y[:, -1, 1].tolist())
        dbp_p.extend(preds[:, -1, 1].tolist())
        map_t.extend(y[:, -1, 2].tolist())
        map_p.extend(preds[:, -1, 2].tolist())

    def _stats(true, pred, name):
        true, pred = np.array(true), np.array(pred)
        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        r, p = pearsonr(true, pred)
        print(f"  {name}: RMSE = {rmse:.2f} mmHg  |  r = {r:.3f}  (p = {p:.2e})")
        return {'rmse': rmse, 'r': float(r), 'p': float(p), 'n': len(true)}

    print(f"\n{'='*50}\n{label.upper()} SET EVALUATION")
    return {
        'sbp': _stats(sbp_t, sbp_p, 'SBP'),
        'dbp': _stats(dbp_t, dbp_p, 'DBP'),
        'map': _stats(map_t, map_p, 'MAP'),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    seq_len = args.seq_len

    print('=' * 60)
    print('LSTM Training -- Radha et al. 2019 Architecture')
    print(f'TensorFlow {tf.__version__}')
    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPUs available: {len(gpus)}')
    if gpus:
        for g in gpus:
            print(f'  {g}')
    print('=' * 60)

    # ── Load ──
    print(f'\nLoading records from {data_dir} ...')
    all_records = load_records(data_dir)
    print(f'  Loaded: {len(all_records)} records')
    if len(all_records) < 10:
        print('Too few records. Run preprocess_mimic.py first.')
        return

    # ── Split by record ──
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(all_records))
    n_train = int(len(idx) * TRAIN_FRAC)
    n_val   = int(len(idx) * VAL_FRAC)
    train_recs = [all_records[i] for i in idx[:n_train]]
    val_recs   = [all_records[i] for i in idx[n_train:n_train + n_val]]
    test_recs  = [all_records[i] for i in idx[n_train + n_val:]]
    print(f'  Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}')

    # ── Normalize ──
    train_recs, feat_mean, feat_std = normalize_features(train_recs)
    val_recs,   _,         _        = normalize_features(val_recs,  feat_mean, feat_std)
    test_recs,  _,         _        = normalize_features(test_recs, feat_mean, feat_std)

    np.save(out_dir / 'feat_mean.npy', feat_mean)
    np.save(out_dir / 'feat_std.npy',  feat_std)
    print(f'  Normalization stats saved -> {out_dir}')

    # ── Build sequences ──
    X_train, y_train = make_sequences(train_recs, seq_len, SEQ_STEP_TRAIN)
    X_val,   y_val   = make_sequences(val_recs,   seq_len, SEQ_STEP_TRAIN)
    print(f'\n  Train sequences: {len(X_train)}')
    print(f'  Val sequences:   {len(X_val)}  (stride {SEQ_STEP_TRAIN}, dense eval kept for final scoring)')

    train_mean_sbp = float(y_train[:, :, 0].mean())
    print(f'  Train mean relative SBP: {train_mean_sbp:.3f} mmHg')

    # ── Build model ──
    model = build_model(seq_len)
    model.summary()

    loss_fn = WeightedMSE(train_mean_sbp=train_mean_sbp, name='weighted_mse')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=loss_fn,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
    )

    # ── Callbacks ──
    ckpt_path = str(out_dir / 'best_model.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_loss',
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out_dir / 'training_log.csv')),
    ]

    # ── Train ──
    print(f'\nTraining (max {args.epochs} epochs | batch {args.batch_size} '
          f'| patience {PATIENCE}) ...\n')
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_min = (time.time() - t0) / 60
    print(f'\nTraining finished in {elapsed_min:.1f} min')
    print(f'Best val loss: '
          f'{min(history.history["val_loss"]):.4f} '
          f'at epoch {np.argmin(history.history["val_loss"]) + 1}')

    # ── Save history ──
    hist = {k: [float(v) for v in vals]
            for k, vals in history.history.items()}
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(hist, f, indent=2)

    # ── Evaluate ──
    print('\nLoading best checkpoint ...')
    model = tf.keras.models.load_model(
        ckpt_path,
        custom_objects={'WeightedMSE': WeightedMSE},
    )
    val_results  = evaluate(model, val_recs,  seq_len, SEQ_STEP_EVAL, 'validation')
    test_results = evaluate(model, test_recs, seq_len, SEQ_STEP_EVAL, 'test')

    eval_out = {
        'n_train_records': len(train_recs),
        'n_val_records':   len(val_recs),
        'n_test_records':  len(test_recs),
        'n_train_seqs':    int(len(X_train)),
        'n_val_seqs':      int(len(X_val)),
        'train_mean_sbp':  train_mean_sbp,
        'seq_len':         seq_len,
        'elapsed_min':     round(elapsed_min, 1),
        'validation':      val_results,
        'test':            test_results,
    }
    with open(out_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_out, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Model:         {ckpt_path}')
    print(f'Norm stats:    {out_dir}/feat_mean.npy + feat_std.npy')
    print(f'Eval results:  {out_dir}/eval_results.json')
    print('='*60)


if __name__ == '__main__':
    main()

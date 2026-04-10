"""
dashboard.py - Generate an interactive HTML dashboard for the seizure detection project.

Usage:
    python dashboard.py

Outputs:
    results/dashboard.html  (open in any browser)
"""

import os
import json
import base64
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support,
)

import config as cfg
from src.data_loader import load_data
from src.preprocess import preprocess
from src.model import SeizureNet
from src.train import train_model
from utils.helpers import set_seed


def _img_to_base64(path):
    """Read an image file and return a base64 data-URI string."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lstrip(".")
    return f"data:image/{ext};base64,{encoded}"


def gather_data():
    """Load model, run evaluation, and collect all dashboard data."""
    set_seed(cfg.RANDOM_STATE)

    # Load & preprocess
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    # DataLoader
    test_ds = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Load model
    model = SeizureNet()
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, weights_only=True))
    model.eval()

    # Predictions
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Per-class
    p_cls, r_cls, f_cls, sup_cls = precision_recall_fscore_support(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Class distribution (full dataset)
    class_counts = [int((y == c).sum()) for c in range(cfg.NUM_CLASSES)]

    # Sample predictions (first 20 from test set)
    sample_indices = list(range(min(20, len(y_true))))
    samples = []
    for i in sample_indices:
        samples.append({
            "features": [round(float(v), 2) for v in X_test[i]],
            "true": int(y_true[i]),
            "pred": int(y_pred[i]),
            "confidence": round(float(y_probs[i].max()) * 100, 1),
            "probs": [round(float(p) * 100, 1) for p in y_probs[i]],
            "correct": bool(y_true[i] == y_pred[i]),
        })

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Feature statistics
    feature_names = [f"X{i+1}" for i in range(cfg.NUM_FEATURES)]
    import pandas as pd
    df = pd.read_csv(cfg.RAW_DATA_PATH)
    feat_means = [round(float(df[f"X{i+1}"].mean()), 2) for i in range(cfg.NUM_FEATURES)]
    feat_stds = [round(float(df[f"X{i+1}"].std()), 2) for i in range(cfg.NUM_FEATURES)]

    return {
        "metrics": {"accuracy": round(acc, 4), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)},
        "per_class": {
            "precision": [round(float(v), 4) for v in p_cls],
            "recall": [round(float(v), 4) for v in r_cls],
            "f1": [round(float(v), 4) for v in f_cls],
            "support": [int(v) for v in sup_cls],
        },
        "confusion_matrix": cm,
        "class_names": cfg.CLASS_NAMES,
        "class_counts": class_counts,
        "num_classes": cfg.NUM_CLASSES,
        "samples": samples,
        "model_info": {
            "architecture": str(model),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "hidden_sizes": cfg.HIDDEN_SIZES,
            "dropout": cfg.DROPOUT_RATE,
            "optimizer": "Adam",
            "lr": cfg.LEARNING_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.NUM_EPOCHS,
            "early_stop": cfg.EARLY_STOP_PATIENCE,
        },
        "dataset_info": {
            "total_samples": int(len(y)),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "num_features": cfg.NUM_FEATURES,
            "feature_names": feature_names,
            "feat_means": feat_means,
            "feat_stds": feat_stds,
        },
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EEG Seizure Detection — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* ───── Reset & Base ───── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0b0f1a;
  --surface: #111827;
  --surface2: #1a2236;
  --border: #1e293b;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --accent: #6366f1;
  --accent-glow: rgba(99,102,241,.25);
  --green: #22c55e;
  --green-dim: rgba(34,197,94,.15);
  --blue: #3b82f6;
  --blue-dim: rgba(59,130,246,.15);
  --amber: #f59e0b;
  --amber-dim: rgba(245,158,11,.15);
  --rose: #f43f5e;
  --rose-dim: rgba(244,63,94,.15);
  --cyan: #06b6d4;
  --purple: #a855f7;
  --radius: 16px;
  --radius-sm: 10px;
  --shadow: 0 4px 24px rgba(0,0,0,.35);
}
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  overflow-x: hidden;
}
a { color: var(--accent); text-decoration: none; }

/* ───── Background Grid ───── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: -1;
  background-image:
    radial-gradient(circle at 20% 20%, rgba(99,102,241,.08) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(6,182,212,.06) 0%, transparent 50%),
    linear-gradient(rgba(255,255,255,.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.02) 1px, transparent 1px);
  background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
}

/* ───── Layout ───── */
.container {
  max-width: 1360px;
  margin: 0 auto;
  padding: 24px;
}

/* ───── Header ───── */
.header {
  text-align: center;
  padding: 48px 24px 36px;
  position: relative;
}
.header::after {
  content: '';
  position: absolute;
  bottom: 0; left: 50%;
  transform: translateX(-50%);
  width: 120px; height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
  border-radius: 2px;
}
.header h1 {
  font-size: 2.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, #fff 0%, var(--accent) 50%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}
.header p {
  color: var(--text-dim);
  font-size: .95rem;
  margin-top: 8px;
}
.badge {
  display: inline-block;
  margin-top: 12px;
  padding: 4px 14px;
  font-size: .75rem;
  font-weight: 600;
  background: var(--accent-glow);
  color: var(--accent);
  border: 1px solid rgba(99,102,241,.3);
  border-radius: 999px;
  letter-spacing: .5px;
  text-transform: uppercase;
}

/* ───── Metric Cards ───── */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin: 36px 0;
}
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: var(--shadow); }
.metric-card .icon {
  width: 44px; height: 44px;
  border-radius: 12px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 1.3rem;
  margin-bottom: 12px;
}
.metric-card .value {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -1px;
}
.metric-card .label {
  font-size: .78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-dim);
  margin-top: 4px;
}
.mc-green .icon  { background: var(--green-dim); color: var(--green); }
.mc-green .value { color: var(--green); }
.mc-blue .icon   { background: var(--blue-dim);  color: var(--blue); }
.mc-blue .value  { color: var(--blue); }
.mc-amber .icon  { background: var(--amber-dim); color: var(--amber); }
.mc-amber .value { color: var(--amber); }
.mc-rose .icon   { background: var(--rose-dim);  color: var(--rose); }
.mc-rose .value  { color: var(--rose); }

/* glow stripe */
.metric-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 3px;
}
.mc-green::before  { background: var(--green); }
.mc-blue::before   { background: var(--blue); }
.mc-amber::before  { background: var(--amber); }
.mc-rose::before   { background: var(--rose); }

/* ───── Panels / Cards ───── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
  margin-bottom: 20px;
  box-shadow: var(--shadow);
}
.panel-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 20px;
  display: flex; align-items: center; gap: 8px;
}
.panel-title .dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent-glow);
}

/* ───── Grid Layouts ───── */
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.grid-3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
}
@media (max-width: 900px) {
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
}

/* ───── Tables ───── */
.table-wrap { overflow-x: auto; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: .88rem;
}
th, td {
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
th {
  font-weight: 600;
  color: var(--text-dim);
  font-size: .76rem;
  text-transform: uppercase;
  letter-spacing: .8px;
}
tr:hover td { background: rgba(255,255,255,.02); }

/* ───── Confusion Matrix (HTML) ───── */
.cm-grid {
  display: grid;
  gap: 3px;
  max-width: 420px;
  margin: 0 auto;
}
.cm-cell {
  display: flex; align-items: center; justify-content: center;
  aspect-ratio: 1;
  border-radius: 8px;
  font-weight: 700;
  font-size: 1rem;
  transition: transform .15s;
}
.cm-cell:hover { transform: scale(1.08); z-index: 1; }
.cm-label {
  display: flex; align-items: center; justify-content: center;
  font-weight: 600;
  font-size: .72rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: .5px;
}

/* ───── Sample Predictions ───── */
.sample-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  transition: background .15s;
}
.sample-row:hover { background: rgba(255,255,255,.03); }
.sample-row + .sample-row { border-top: 1px solid var(--border); }
.pill {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: .72rem;
  font-weight: 600;
  white-space: nowrap;
}
.pill-correct { background: var(--green-dim); color: var(--green); }
.pill-wrong   { background: var(--rose-dim);  color: var(--rose); }
.conf-bar {
  height: 6px;
  border-radius: 3px;
  background: var(--border);
  flex: 1;
  overflow: hidden;
}
.conf-bar-inner { height: 100%; border-radius: 3px; }

/* ───── Architecture ───── */
.arch-block {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px 20px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: .8rem;
  color: var(--text-dim);
  white-space: pre-wrap;
  overflow-x: auto;
  line-height: 1.7;
}

/* ───── Info Pills Row ───── */
.info-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
}
.info-pill {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 6px 16px;
  font-size: .78rem;
  display: flex; gap: 6px;
}
.info-pill .ip-label { color: var(--text-dim); }
.info-pill .ip-value { font-weight: 600; color: var(--text); }

/* ───── Section Dividers ───── */
.section-title {
  font-size: 1.1rem;
  font-weight: 700;
  margin: 40px 0 16px;
  padding-left: 14px;
  border-left: 3px solid var(--accent);
}

/* ───── Tabs ───── */
.tab-bar {
  display: flex; gap: 4px;
  margin-bottom: 20px;
  background: var(--bg);
  border-radius: var(--radius-sm);
  padding: 4px;
  border: 1px solid var(--border);
}
.tab-btn {
  flex: 1;
  padding: 8px 12px;
  border: none;
  background: transparent;
  color: var(--text-dim);
  font-family: inherit;
  font-size: .82rem;
  font-weight: 600;
  border-radius: 8px;
  cursor: pointer;
  transition: all .2s;
}
.tab-btn.active {
  background: var(--accent);
  color: #fff;
  box-shadow: 0 2px 12px var(--accent-glow);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ───── Footer ───── */
.footer {
  text-align: center;
  padding: 32px 0;
  color: var(--text-dim);
  font-size: .78rem;
  border-top: 1px solid var(--border);
  margin-top: 40px;
}
</style>
</head>
<body>

<!-- ═══════ HEADER ═══════ -->
<div class="header">
  <h1>⚡ EEG Seizure Detection</h1>
  <p>Deep Learning Classification on the BEED Dataset · PyTorch MLP</p>
  <span class="badge">Multi-class · 4 EEG States</span>
</div>

<div class="container">

<!-- ═══════ METRIC CARDS ═══════ -->
<div class="metrics-grid" id="metricsGrid"></div>

<!-- ═══════ DATASET & MODEL INFO ═══════ -->
<div class="info-pills" id="infoPills"></div>

<!-- ═══════ CHARTS ROW ═══════ -->
<div class="section-title">Training Performance</div>
<div class="grid-2">
  <div class="panel">
    <div class="panel-title"><span class="dot"></span>Confusion Matrix</div>
    <div id="cmContainer"></div>
    <div style="display:flex; justify-content:center; gap:16px; margin-top:16px; flex-wrap:wrap;" id="cmLegend"></div>
  </div>
  <div class="panel">
    <div class="panel-title"><span class="dot"></span>Class Distribution</div>
    <canvas id="classDistChart" height="280"></canvas>
  </div>
</div>

<!-- ═══════ PER-CLASS METRICS ═══════ -->
<div class="section-title">Per-Class Performance</div>
<div class="panel">
  <div class="panel-title"><span class="dot"></span>Classification Report</div>
  <div class="table-wrap">
    <table id="classTable">
      <thead>
        <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th><th>Visual</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<!-- ═══════ FEATURE ANALYSIS ═══════ -->
<div class="section-title">Feature Statistics</div>
<div class="panel">
  <div class="panel-title"><span class="dot"></span>EEG Channel Analysis</div>
  <canvas id="featureChart" height="180"></canvas>
</div>

<!-- ═══════ SAMPLE PREDICTIONS ═══════ -->
<div class="section-title">Sample Predictions</div>
<div class="panel">
  <div class="panel-title"><span class="dot"></span>Test Set Predictions</div>
  <div class="tab-bar">
    <button class="tab-btn active" onclick="filterSamples('all')">All</button>
    <button class="tab-btn" onclick="filterSamples('correct')">✓ Correct</button>
    <button class="tab-btn" onclick="filterSamples('wrong')">✗ Wrong</button>
  </div>
  <div id="samplesContainer"></div>
</div>

<!-- ═══════ MODEL ARCHITECTURE ═══════ -->
<div class="section-title">Model Architecture</div>
<div class="panel">
  <div class="panel-title"><span class="dot"></span>SeizureNet (MLP)</div>
  <div id="archViz"></div>
  <div class="arch-block" id="archBlock" style="margin-top:16px;"></div>
</div>

<!-- ═══════ FOOTER ═══════ -->
<div class="footer">
  EEG Seizure Detection Dashboard &middot; Built with PyTorch &amp; Chart.js &middot; BEED Dataset
</div>

</div><!-- /container -->

<script>
// ═══════ DATA (injected by Python) ═══════
const DATA = __DATA_PLACEHOLDER__;

// ═══════ COLORS ═══════
const CLASS_COLORS = ['#6366f1', '#3b82f6', '#f59e0b', '#f43f5e'];
const CLASS_COLORS_DIM = ['rgba(99,102,241,.2)','rgba(59,130,246,.2)','rgba(245,158,11,.2)','rgba(244,63,94,.2)'];

// ═══════ METRIC CARDS ═══════
(function renderMetrics() {
  const grid = document.getElementById('metricsGrid');
  const items = [
    { key:'accuracy',  icon:'🎯', cls:'mc-green', label:'Accuracy' },
    { key:'precision', icon:'🔬', cls:'mc-blue',  label:'Precision' },
    { key:'recall',    icon:'📡', cls:'mc-amber', label:'Recall' },
    { key:'f1',        icon:'⚖️', cls:'mc-rose',  label:'F1 Score' },
  ];
  items.forEach(it => {
    const v = (DATA.metrics[it.key] * 100).toFixed(1);
    grid.innerHTML += `
      <div class="metric-card ${it.cls}">
        <div class="icon">${it.icon}</div>
        <div class="value">${v}%</div>
        <div class="label">${it.label}</div>
      </div>`;
  });
})();

// ═══════ INFO PILLS ═══════
(function renderInfoPills() {
  const c = document.getElementById('infoPills');
  const ds = DATA.dataset_info, mi = DATA.model_info;
  const pills = [
    ['Samples', ds.total_samples.toLocaleString()],
    ['Features', ds.num_features],
    ['Classes', DATA.num_classes],
    ['Train / Test', `${ds.train_samples} / ${ds.test_samples}`],
    ['Parameters', mi.total_params.toLocaleString()],
    ['Optimizer', `${mi.optimizer} (lr=${mi.lr})`],
    ['Batch', mi.batch_size],
    ['Epochs', mi.epochs],
    ['Early Stop', `${mi.early_stop} patience`],
    ['Dropout', mi.dropout],
  ];
  pills.forEach(([l,v]) => {
    c.innerHTML += `<div class="info-pill"><span class="ip-label">${l}</span><span class="ip-value">${v}</span></div>`;
  });
})();

// ═══════ CONFUSION MATRIX ═══════
(function renderCM() {
  const cm = DATA.confusion_matrix;
  const n = cm.length;
  const maxVal = Math.max(...cm.flat());
  const container = document.getElementById('cmContainer');

  const sz = n + 1; // +1 for labels
  let html = `<div class="cm-grid" style="grid-template-columns: 60px repeat(${n}, 1fr); grid-template-rows: repeat(${n}, 1fr) 36px;">`;

  // Data cells
  for (let i = 0; i < n; i++) {
    // Row label
    // We place them outside; let's do a simpler approach
    for (let j = 0; j < n; j++) {
      const val = cm[i][j];
      const intensity = val / maxVal;
      const bg = i === j
        ? `rgba(99,102,241,${0.15 + intensity * 0.7})`
        : `rgba(244,63,94,${intensity * 0.5})`;
      const color = intensity > 0.5 ? '#fff' : 'var(--text-dim)';
      html += `<div class="cm-cell" style="background:${bg};color:${color};grid-column:${j+2};grid-row:${i+1}">${val}</div>`;
    }
  }

  // Row labels (left)
  for (let i = 0; i < n; i++) {
    html += `<div class="cm-label" style="grid-column:1;grid-row:${i+1}; font-size:.68rem;">${DATA.class_names[i]}</div>`;
  }
  // Col labels (bottom)
  for (let j = 0; j < n; j++) {
    html += `<div class="cm-label" style="grid-column:${j+2};grid-row:${n+1}">${DATA.class_names[j]}</div>`;
  }

  html += '</div>';
  html += '<div style="text-align:center;margin-top:8px;font-size:.72rem;color:var(--text-dim)">Rows = True · Columns = Predicted</div>';
  container.innerHTML = html;
})();

// ═══════ CLASS DISTRIBUTION CHART ═══════
(function renderClassDist() {
  new Chart(document.getElementById('classDistChart'), {
    type: 'doughnut',
    data: {
      labels: DATA.class_names,
      datasets: [{
        data: DATA.class_counts,
        backgroundColor: CLASS_COLORS,
        borderColor: 'var(--surface)',
        borderWidth: 3,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 }, padding: 16, usePointStyle: true, pointStyleWidth: 12 }
        },
        tooltip: {
          backgroundColor: '#1a2236',
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          borderColor: '#1e293b',
          borderWidth: 1,
          cornerRadius: 10,
          padding: 12,
        }
      }
    }
  });
})();

// ═══════ PER-CLASS TABLE ═══════
(function renderClassTable() {
  const tbody = document.querySelector('#classTable tbody');
  const pc = DATA.per_class;
  for (let i = 0; i < DATA.num_classes; i++) {
    const f1 = pc.f1[i];
    const barW = (f1 * 100).toFixed(0);
    tbody.innerHTML += `
      <tr>
        <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${CLASS_COLORS[i]};margin-right:8px"></span>${DATA.class_names[i]}</td>
        <td>${(pc.precision[i]*100).toFixed(1)}%</td>
        <td>${(pc.recall[i]*100).toFixed(1)}%</td>
        <td><strong>${(pc.f1[i]*100).toFixed(1)}%</strong></td>
        <td>${pc.support[i]}</td>
        <td>
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden">
              <div style="width:${barW}%;height:100%;background:${CLASS_COLORS[i]};border-radius:4px"></div>
            </div>
          </div>
        </td>
      </tr>`;
  }
})();

// ═══════ FEATURE CHART ═══════
(function renderFeatureChart() {
  const ds = DATA.dataset_info;
  new Chart(document.getElementById('featureChart'), {
    type: 'bar',
    data: {
      labels: ds.feature_names,
      datasets: [
        {
          label: 'Mean',
          data: ds.feat_means,
          backgroundColor: 'rgba(99,102,241,.6)',
          borderColor: '#6366f1',
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: 'Std Dev',
          data: ds.feat_stds,
          backgroundColor: 'rgba(6,182,212,.5)',
          borderColor: '#06b6d4',
          borderWidth: 1,
          borderRadius: 4,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 }, usePointStyle: true, pointStyleWidth: 10 }
        },
        tooltip: {
          backgroundColor: '#1a2236', titleColor: '#e2e8f0', bodyColor: '#94a3b8',
          borderColor: '#1e293b', borderWidth: 1, cornerRadius: 10, padding: 12,
        }
      },
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,.04)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,.04)' } },
      }
    }
  });
})();

// ═══════ SAMPLE PREDICTIONS ═══════
let allSamplesHTML = '';
(function renderSamples() {
  const c = document.getElementById('samplesContainer');
  DATA.samples.forEach((s, idx) => {
    const cls = s.correct ? 'pill-correct' : 'pill-wrong';
    const icon = s.correct ? '✓' : '✗';
    const barColor = s.correct ? 'var(--green)' : 'var(--rose)';
    const trueName = DATA.class_names[s.true];
    const predName = DATA.class_names[s.pred];
    allSamplesHTML += `
      <div class="sample-row" data-correct="${s.correct}">
        <span style="color:var(--text-dim);font-size:.75rem;width:28px">#${idx+1}</span>
        <span class="pill ${cls}">${icon} ${predName}</span>
        <span style="color:var(--text-dim);font-size:.78rem;">True: <strong style="color:var(--text)">${trueName}</strong></span>
        <div class="conf-bar">
          <div class="conf-bar-inner" style="width:${s.confidence}%;background:${barColor}"></div>
        </div>
        <span style="font-size:.78rem;font-weight:600;min-width:48px;text-align:right">${s.confidence}%</span>
      </div>`;
  });
  c.innerHTML = allSamplesHTML;
})();

function filterSamples(type) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  const c = document.getElementById('samplesContainer');
  c.innerHTML = allSamplesHTML;
  if (type !== 'all') {
    const show = type === 'correct';
    c.querySelectorAll('.sample-row').forEach(row => {
      if ((row.dataset.correct === 'true') !== show) row.style.display = 'none';
    });
  }
}

// ═══════ MODEL ARCHITECTURE VIZ ═══════
(function renderArch() {
  const mi = DATA.model_info;
  const layers = [DATA.dataset_info.num_features, ...mi.hidden_sizes, DATA.num_classes];
  const labels = ['Input', ...mi.hidden_sizes.map(h => `Dense(${h})\nBN+ReLU+Drop`), `Output(${DATA.num_classes})`];
  const colors = ['var(--cyan)', ...mi.hidden_sizes.map(() => 'var(--accent)'), 'var(--green)'];

  let html = '<div style="display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap;padding:12px 0">';
  layers.forEach((size, i) => {
    const h = Math.max(40, Math.min(120, size * 1.2));
    html += `<div style="display:flex;flex-direction:column;align-items:center;gap:4px">
      <div style="width:52px;height:${h}px;background:${colors[i]};opacity:.7;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;color:#fff">${size}</div>
      <div style="font-size:.65rem;color:var(--text-dim);text-align:center;white-space:pre-line;line-height:1.3">${labels[i]}</div>
    </div>`;
    if (i < layers.length - 1) {
      html += '<div style="color:var(--text-dim);font-size:1.2rem">→</div>';
    }
  });
  html += '</div>';
  document.getElementById('archViz').innerHTML = html;
  document.getElementById('archBlock').textContent = mi.architecture;
})();
</script>
</body>
</html>"""


def generate_dashboard():
    """Generate the HTML dashboard file."""
    print("Gathering model data...")
    data = gather_data()

    print("Building dashboard HTML...")
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(data))

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.RESULTS_DIR, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ Dashboard saved to: {out_path}")
    print(f"  Open in your browser to view!")
    return out_path


if __name__ == "__main__":
    generate_dashboard()

"""
FastAPI Drift Monitoring Demo

Run with: python examples/fastapi_demo.py
Open: http://localhost:8000
"""

import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from driftwatch import Monitor

# =============================================================================
# 1. SHARED STATE (Global - accessible by both middleware and routes)
# =============================================================================

class SharedDriftState:
    """Global state for drift monitoring."""
    
    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.last_report = None
        self.last_check_time = None
        self.request_count = 0

# Global instance
DRIFT_STATE = SharedDriftState()

# =============================================================================
# 2. Reference Data & Monitor
# =============================================================================

np.random.seed(42)
reference_data = pd.DataFrame({
    "age": np.random.normal(35, 10, 1000).clip(18, 80),
    "income": np.random.lognormal(10.5, 0.5, 1000),
    "credit_score": np.random.normal(700, 50, 1000).clip(300, 850),
    "loan_amount": np.random.lognormal(9, 0.8, 1000),
})

monitor = Monitor(
    reference_data=reference_data,
    thresholds={"psi": 0.15, "ks_pvalue": 0.05}
)

MONITORED_FEATURES = ["age", "income", "credit_score", "loan_amount"]
MIN_SAMPLES = 5

# =============================================================================
# 3. FastAPI App
# =============================================================================

app = FastAPI(title="DriftWatch Demo")


@app.post("/predict")
async def predict(age: float, income: float, credit_score: float, loan_amount: float):
    """Prediction endpoint that also collects samples."""
    
    # Store sample in global state
    DRIFT_STATE.samples.append({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
    })
    DRIFT_STATE.request_count += 1
    
    # Dummy prediction
    score = 0.5
    score += 0.1 if credit_score > 700 else -0.1
    score += 0.1 if income > 50000 else -0.1
    approval_prob = max(0, min(1, score + random.uniform(-0.1, 0.1)))
    
    return {
        "approval_probability": round(approval_prob, 3),
        "decision": "APPROVED" if approval_prob > 0.6 else "DENIED",
    }


@app.get("/drift/status")
async def drift_status():
    """Get current drift status."""
    if DRIFT_STATE.last_report is None:
        return {
            "status": "NO_DATA",
            "samples_collected": len(DRIFT_STATE.samples),
            "total_requests": DRIFT_STATE.request_count,
        }
    
    return {
        "status": DRIFT_STATE.last_report.status.value,
        "has_drift": DRIFT_STATE.last_report.has_drift(),
        "drift_ratio": DRIFT_STATE.last_report.drift_ratio(),
        "drifted_features": DRIFT_STATE.last_report.drifted_features(),
        "last_check": DRIFT_STATE.last_check_time.isoformat() if DRIFT_STATE.last_check_time else None,
        "samples_collected": len(DRIFT_STATE.samples),
        "total_requests": DRIFT_STATE.request_count,
    }


@app.get("/drift/report")
async def drift_report():
    """Get full drift report."""
    if DRIFT_STATE.last_report is None:
        return {"error": "No report yet", "samples_collected": len(DRIFT_STATE.samples)}
    return DRIFT_STATE.last_report.to_dict()


@app.post("/drift/check")
async def trigger_check():
    """Manually trigger drift check."""
    if len(DRIFT_STATE.samples) < MIN_SAMPLES:
        return {"error": f"Not enough samples. Need {MIN_SAMPLES}, have {len(DRIFT_STATE.samples)}"}
    
    production_df = pd.DataFrame(DRIFT_STATE.samples)
    report = monitor.check(production_df)
    
    DRIFT_STATE.last_report = report
    DRIFT_STATE.last_check_time = datetime.now(timezone.utc)
    
    return {
        "status": report.status.value,
        "has_drift": report.has_drift(),
        "drift_ratio": report.drift_ratio(),
        "drifted_features": report.drifted_features(),
    }


@app.post("/drift/reset")
async def reset_samples():
    """Reset collected samples."""
    DRIFT_STATE.samples.clear()
    DRIFT_STATE.request_count = 0
    DRIFT_STATE.last_report = None
    return {"message": "Reset complete"}


# =============================================================================
# 4. Dashboard HTML
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DriftWatch Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --card: rgba(30, 41, 59, 0.8);
            --accent: #38bdf8;
            --accent2: #818cf8;
            --success: #34d399;
            --warning: #fbbf24;
            --critical: #f87171;
            --text: #f8fafc;
            --muted: #94a3b8;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(ellipse at top, #1e1b4b, #0f172a);
            min-height: 100vh;
            color: var(--text);
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        h1 {
            font-size: 1.8rem;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .live-badge {
            background: rgba(52, 211, 153, 0.2);
            border: 1px solid var(--success);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            color: var(--success);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .live-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        
        .grid { display: grid; grid-template-columns: 280px 1fr; gap: 2rem; }
        
        .sidebar { display: flex; flex-direction: column; gap: 1rem; }
        
        .panel {
            background: var(--card);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(20px);
        }
        
        .panel-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--muted);
            margin-bottom: 1rem;
        }
        
        button {
            width: 100%;
            padding: 0.9rem;
            border: none;
            border-radius: 10px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .btn-normal {
            background: linear-gradient(135deg, var(--accent), #0284c7);
            color: white;
        }
        .btn-drift {
            background: linear-gradient(135deg, var(--warning), #d97706);
            color: #1a1a1a;
        }
        .btn-action {
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
        }
        button:hover { transform: translateY(-2px); opacity: 0.95; }
        button:active { transform: scale(0.98); }
        
        .main { display: flex; flex-direction: column; gap: 1.5rem; }
        
        .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
        
        .kpi {
            background: var(--card);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 14px;
            padding: 1.2rem;
        }
        .kpi-label { color: var(--muted); font-size: 0.8rem; margin-bottom: 0.3rem; }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; color: var(--muted); font-weight: 500; padding: 0.8rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem; }
        td { padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
        
        .badge {
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
        }
        .badge.drift { background: rgba(248, 113, 113, 0.2); color: var(--critical); }
        .badge.ok { background: rgba(52, 211, 153, 0.2); color: var(--success); }
        
        .log { max-height: 150px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--muted); }
        .log-entry { margin-bottom: 0.2rem; }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 DriftWatch Monitor</h1>
            <div class="live-badge"><div class="live-dot"></div> Live</div>
        </header>
        
        <div class="grid">
            <div class="sidebar">
                <div class="panel">
                    <div class="panel-title">Send Data</div>
                    <button class="btn-normal" onclick="send(10, false)">📤 10 Normal Samples</button>
                    <button class="btn-drift" onclick="send(10, true)">⚠️ 10 Drifted Samples</button>
                </div>
                <div class="panel">
                    <div class="panel-title">Actions</div>
                    <button class="btn-action" onclick="check()">🔍 Run Drift Check</button>
                    <button class="btn-action" onclick="reset()">🔄 Reset Buffer</button>
                </div>
                <div class="panel">
                    <div class="panel-title">Log</div>
                    <div class="log" id="log"></div>
                </div>
            </div>
            
            <div class="main">
                <div class="kpis">
                    <div class="kpi">
                        <div class="kpi-label">STATUS</div>
                        <div class="kpi-value" id="status" style="color: var(--muted);">WAITING</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-label">SAMPLES</div>
                        <div class="kpi-value" id="samples">0</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-label">DRIFT RATIO</div>
                        <div class="kpi-value" id="ratio">0%</div>
                    </div>
                </div>
                
                <div class="panel">
                    <div class="panel-title">Feature Analysis</div>
                    <table>
                        <thead><tr><th>Feature</th><th>Method</th><th>Score</th><th>Threshold</th><th>Status</th></tr></thead>
                        <tbody id="tbody">
                            <tr><td colspan="5" style="text-align:center; color: var(--muted); padding: 2rem;">
                                Send at least 5 samples, then click "Run Drift Check"
                            </td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const log = (msg) => {
            const el = document.getElementById('log');
            el.innerHTML = `<div class="log-entry">${new Date().toLocaleTimeString()} ${msg}</div>` + el.innerHTML;
        };
        
        const update = async () => {
            const res = await fetch('/drift/status');
            const d = await res.json();
            
            const statusEl = document.getElementById('status');
            statusEl.textContent = d.status;
            statusEl.style.color = { OK: 'var(--success)', WARNING: 'var(--warning)', CRITICAL: 'var(--critical)', NO_DATA: 'var(--muted)' }[d.status] || 'var(--text)';
            
            document.getElementById('samples').textContent = d.samples_collected;
            document.getElementById('ratio').textContent = d.drift_ratio ? (d.drift_ratio * 100).toFixed(0) + '%' : '0%';
            
            if (d.status !== 'NO_DATA') {
                const rep = await fetch('/drift/report');
                const r = await rep.json();
                if (r.feature_results) {
                    document.getElementById('tbody').innerHTML = r.feature_results.map(f => `
                        <tr>
                            <td style="color: white; font-weight: 600;">${f.feature_name}</td>
                            <td>${f.method}</td>
                            <td>${f.score.toFixed(4)}</td>
                            <td style="opacity:0.5">${f.threshold}</td>
                            <td><span class="badge ${f.has_drift ? 'drift' : 'ok'}">${f.has_drift ? 'DRIFT' : 'OK'}</span></td>
                        </tr>
                    `).join('');
                }
            }
        };
        
        const send = async (n, drifted) => {
            log(`Sending ${n} ${drifted ? 'drifted' : 'normal'} samples...`);
            for (let i = 0; i < n; i++) {
                const d = drifted
                    ? { age: 65 + Math.random() * 10, income: 150000 + Math.random() * 50000, credit_score: 500 + Math.random() * 100, loan_amount: 200000 + Math.random() * 50000 }
                    : { age: 35 + Math.random() * 10, income: 45000 + Math.random() * 10000, credit_score: 700 + Math.random() * 50, loan_amount: 10000 + Math.random() * 5000 };
                await fetch('/predict?' + new URLSearchParams(d), { method: 'POST' });
            }
            log('Done!');
            update();
        };
        
        const check = async () => {
            log('Running drift check...');
            const res = await fetch('/drift/check', { method: 'POST' });
            const d = await res.json();
            if (d.error) { log('⚠️ ' + d.error); }
            else { log('Result: ' + d.status + (d.has_drift ? ' (Drift detected!)' : '')); }
            update();
        };
        
        const reset = async () => {
            await fetch('/drift/reset', { method: 'POST' });
            log('Buffer reset.');
            document.getElementById('tbody').innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--muted); padding: 2rem;">Buffer cleared.</td></tr>';
            update();
        };
        
        update();
        setInterval(update, 2000);
        log('Ready.');
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


if __name__ == "__main__":
    print("🚀 DriftWatch Demo: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

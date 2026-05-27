"""
Rapha Enterprise Node — Database Layer

Manages:
  - Mock EHR patient data for training
  - Job tracking for async training pipelines
  - Dataset catalog metadata
"""

import os
import sqlite3
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger("rapha.db")

DB_PATH = os.getenv("RAPHA_DB_PATH", "/tmp/rapha/hospital_mock.db")


def get_connection():
    """Get a SQLite connection with row factory enabled."""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all database tables and seed data."""
    conn = get_connection()
    c = conn.cursor()

    # ── EHR Data (legacy) ─────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS ehr_data (
            patient_id TEXT PRIMARY KEY,
            blood_pressure_sys INTEGER,
            blood_pressure_dia INTEGER,
            heart_rate INTEGER
        )
    """)

    # Seed EHR data if empty
    c.execute("SELECT COUNT(*) FROM ehr_data")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO ehr_data VALUES (?, ?, ?, ?)", [
            ('p1', 120, 80, 72),
            ('p2', 130, 85, 75),
            ('p3', 115, 75, 68),
            ('p4', 140, 90, 80),
            ('p5', 125, 82, 70),
        ])

    # ── Training Jobs ─────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'queued',
            dataset_id TEXT,
            model_format TEXT,
            model_architecture TEXT,
            epochs INTEGER DEFAULT 5,
            learning_rate REAL DEFAULT 0.01,
            batch_size INTEGER DEFAULT 32,
            created_at REAL,
            started_at REAL,
            completed_at REAL,
            metrics_json TEXT,
            zk_proof TEXT,
            updated_weights TEXT,
            error TEXT,
            org TEXT
        )
    """)

    # ── Training Logs ─────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            message TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES training_jobs(job_id)
        )
    """)

    # ── Dataset Catalog ───────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS dataset_catalog (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            condition TEXT,
            record_count INTEGER,
            schema_json TEXT,
            data_types_json TEXT,
            created_at TEXT
        )
    """)

    # Seed dataset catalog if empty
    c.execute("SELECT COUNT(*) FROM dataset_catalog")
    if c.fetchone()[0] == 0:
        datasets = [
            (
                "hospital_vitals_v1",
                "Hospital Vitals Dataset v1",
                "Anonymized blood pressure and heart rate readings. The core EHR dataset behind this node.",
                "cardiovascular",
                5,
                json.dumps([
                    {"field": "blood_pressure_sys", "type": "integer", "unit": "mmHg"},
                    {"field": "blood_pressure_dia", "type": "integer", "unit": "mmHg"},
                    {"field": "heart_rate", "type": "integer", "unit": "bpm"},
                ]),
                json.dumps(["vitals", "blood_pressure"]),
                "2026-01-15T00:00:00Z",
            ),
        ]
        c.executemany(
            "INSERT INTO dataset_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            datasets,
        )

    conn.commit()
    conn.close()
    logger.info("Database initialized with all tables.")


# ── EHR Data Queries ──────────────────────────────────────

def get_training_data():
    """Fetch local hospital EHR data for training. Data NEVER leaves this node."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT blood_pressure_sys, blood_pressure_dia, heart_rate FROM ehr_data")
    data = c.fetchall()
    conn.close()
    return data


# ── Job Management ────────────────────────────────────────

def create_job(
    job_id: str,
    dataset_id: str,
    model_format: str,
    model_architecture: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    org: str = "",
) -> dict:
    """Register a new training job."""
    conn = get_connection()
    c = conn.cursor()
    now = time.time()
    c.execute(
        """INSERT OR REPLACE INTO training_jobs
           (job_id, status, dataset_id, model_format, model_architecture,
            epochs, learning_rate, batch_size, created_at, org)
           VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)""",
        (job_id, dataset_id, model_format, model_architecture,
         epochs, learning_rate, batch_size, now, org),
    )
    conn.commit()
    conn.close()
    return {"job_id": job_id, "status": "queued"}


def update_job_status(
    job_id: str,
    status: str,
    metrics: dict | None = None,
    zk_proof: str | None = None,
    updated_weights: str | None = None,
    error: str | None = None,
):
    """Update job status and results."""
    conn = get_connection()
    c = conn.cursor()

    updates = ["status = ?"]
    values = [status]

    if status == "running":
        updates.append("started_at = ?")
        values.append(time.time())
    elif status in ("completed", "failed"):
        updates.append("completed_at = ?")
        values.append(time.time())

    if metrics:
        updates.append("metrics_json = ?")
        values.append(json.dumps(metrics))
    if zk_proof:
        updates.append("zk_proof = ?")
        values.append(zk_proof)
    if updated_weights:
        updates.append("updated_weights = ?")
        values.append(updated_weights)
    if error:
        updates.append("error = ?")
        values.append(error)

    values.append(job_id)
    c.execute(f"UPDATE training_jobs SET {', '.join(updates)} WHERE job_id = ?", values)
    conn.commit()
    conn.close()


def get_job(job_id: str) -> dict | None:
    """Fetch a job by ID."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    if result.get("metrics_json"):
        result["metrics"] = json.loads(result["metrics_json"])
    else:
        result["metrics"] = {}
    return result


# ── Training Logs ─────────────────────────────────────────

def append_log(job_id: str, message: str):
    """Append a log line for a training job."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO training_logs (job_id, timestamp, message) VALUES (?, ?, ?)",
        (job_id, time.time(), message),
    )
    conn.commit()
    conn.close()


def get_logs(job_id: str, offset: int = 0) -> list[str]:
    """Fetch log lines for a job, starting from offset."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT message FROM training_logs WHERE job_id = ? ORDER BY id LIMIT -1 OFFSET ?",
        (job_id, offset),
    )
    rows = c.fetchall()
    conn.close()
    return [row["message"] for row in rows]


# ── Dataset Catalog ───────────────────────────────────────

def list_datasets(condition: str | None = None) -> list[dict]:
    """List available datasets, optionally filtered by condition."""
    conn = get_connection()
    c = conn.cursor()
    if condition:
        c.execute(
            "SELECT * FROM dataset_catalog WHERE condition LIKE ?",
            (f"%{condition}%",),
        )
    else:
        c.execute("SELECT * FROM dataset_catalog")
    rows = c.fetchall()
    conn.close()

    import os
    node_id = os.getenv("RAPHA_NODE_ID", "node-unnamed")

    datasets = []
    for row in rows:
        d = dict(row)
        d["node_id"] = node_id
        d["node_url"] = ""  # Filled by caller
        d["schema"] = json.loads(d.pop("schema_json", "[]"))
        d["data_types"] = json.loads(d.pop("data_types_json", "[]"))
        datasets.append(d)
    return datasets


def get_dataset(dataset_id: str) -> dict | None:
    """Get a single dataset by ID."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM dataset_catalog WHERE id = ?", (dataset_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None

    import os
    d = dict(row)
    d["node_id"] = os.getenv("RAPHA_NODE_ID", "node-unnamed")
    d["schema"] = json.loads(d.pop("schema_json", "[]"))
    d["data_types"] = json.loads(d.pop("data_types_json", "[]"))
    return d


if __name__ == '__main__':
    init_db()

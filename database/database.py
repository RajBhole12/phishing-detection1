"""
database.py
------------
SQLite persistence layer for scan history. Uses parameterized queries
throughout to avoid SQL injection, and a context-managed connection
per call to keep things simple and thread-safe for Flask's dev server.
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "scans.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            risk_score REAL NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            scanned_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_scan(url: str, prediction: str, risk_score: float, confidence: float, model_used: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO scans (url, prediction, risk_score, confidence, model_used, scanned_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (url, prediction, risk_score, confidence, model_used,
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def get_history(search: str = None, prediction_filter: str = None,
                 sort_by: str = "scanned_at", sort_dir: str = "desc", limit: int = 500):
    allowed_sort_cols = {"scanned_at", "risk_score", "confidence", "url", "prediction"}
    if sort_by not in allowed_sort_cols:
        sort_by = "scanned_at"
    sort_dir = "DESC" if sort_dir.lower() != "asc" else "ASC"

    query = "SELECT * FROM scans WHERE 1=1"
    params = []
    if search:
        query += " AND url LIKE ?"
        params.append(f"%{search}%")
    if prediction_filter and prediction_filter.lower() != "all":
        query += " AND prediction = ?"
        params.append(prediction_filter)

    query += f" ORDER BY {sort_by} {sort_dir} LIMIT ?"
    params.append(limit)

    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history():
    conn = get_connection()
    conn.execute("DELETE FROM scans")
    conn.commit()
    conn.close()


def get_statistics():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
    phishing = conn.execute("SELECT COUNT(*) c FROM scans WHERE prediction='Phishing'").fetchone()["c"]
    suspicious = conn.execute("SELECT COUNT(*) c FROM scans WHERE prediction='Suspicious'").fetchone()["c"]
    legitimate = conn.execute("SELECT COUNT(*) c FROM scans WHERE prediction='Legitimate'").fetchone()["c"]
    recent = conn.execute("SELECT * FROM scans ORDER BY scanned_at DESC LIMIT 10").fetchall()
    activity = conn.execute("""
        SELECT substr(scanned_at, 1, 10) AS day, COUNT(*) AS count
        FROM scans GROUP BY day ORDER BY day DESC LIMIT 14
    """).fetchall()
    conn.close()

    return {
        "total_scans": total,
        "phishing_detected": phishing,
        "suspicious_urls": suspicious,
        "legitimate_urls": legitimate,
        "recent_scans": [dict(r) for r in recent],
        "activity_over_time": [dict(r) for r in activity][::-1],
    }

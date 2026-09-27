"""
Storage layer. Plain sqlite3 (stdlib) - deliberately no ORM. This is a thin
slice; a raw SQL table is easier for a reviewer to read in five seconds than
an ORM model definition would be.

Schema choices (worth defending in the design note):
- We store the FULL probability vector (all_class_probabilities), not just
  the winning class + its confidence. Storage cost is negligible (a few
  hundred bytes of JSON per row) and it's the difference between being able
  to answer "was Forest a close second guess?" later vs. not.
- model_version is stored per-row, not just once globally, because in a
  real deployment the model WILL be swapped/retrained over time, and rows
  from different model versions should never be silently treated as
  comparable when an analyst queries historical results.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    all_class_probabilities TEXT NOT NULL,   -- JSON: {"Forest": 0.81, ...}
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def insert_result(
    filename: str,
    predicted_class: str,
    confidence: float,
    all_class_probabilities: dict,
    model_version: str,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO classifications
            (filename, predicted_class, confidence, all_class_probabilities,
             model_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            predicted_class,
            confidence,
            json.dumps(all_class_probabilities),
            model_version,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id
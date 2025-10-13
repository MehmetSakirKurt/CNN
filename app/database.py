"""Thin database layer for storing patients and inference results (PostgreSQL)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Dict, Iterator, List, Protocol, Sequence

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError as exc:  # pragma: no cover - reported at runtime
    raise ImportError(
        "psycopg2 is required for database connectivity. Install with "
        "`pip install psycopg2-binary`."
    ) from exc


@dataclass
class PatientInfo:
    """Domain object describing a single patient."""

    national_id: str
    first_name: str
    last_name: str
    birth_place: str | None = None
    birth_date: date | None = None
    gender: str | None = None



class DatabaseBackend(Protocol):
    """Common behaviour shared by concrete database adapters."""

    def upsert_patient(self, patient: PatientInfo) -> str: ...

    def list_patients(self) -> List[Dict]: ...

    def record_exam_result(
        self,
        patient_id: str,
        *,
        image_path: str,
        prediction: Dict,
        model_version: str,
    ) -> str: ...

    def patient_history(self, patient_id: str, limit: int = 50) -> List[Dict]: ...

    def recent_results(self, limit: int = 25) -> List[Dict]: ...

    def close(self) -> None: ...


class InMemoryDatabase(DatabaseBackend):
    """Ephemeral store used when a PostgreSQL DSN is not provided."""

    is_ephemeral = True

    def __init__(self) -> None:
        self._patients: Dict[str, Dict] = {}
        self._patients_by_national_id: Dict[str, str] = {}
        self._exam_results: List[Dict] = []

    def upsert_patient(self, patient: PatientInfo) -> str:
        payload = asdict(patient)
        now = datetime.utcnow()
        existing_id = self._patients_by_national_id.get(patient.national_id)
        if existing_id:
            record = self._patients[existing_id]
            record.update({**payload, "updated_at": now})
            return existing_id

        patient_id = str(uuid.uuid4())
        record = {
            "patient_id": patient_id,
            "created_at": now,
            **payload,
        }
        self._patients[patient_id] = record
        self._patients_by_national_id[patient.national_id] = patient_id
        return patient_id

    def list_patients(self) -> List[Dict]:
        def latest_result(pid: str) -> Dict | None:
            results = [res for res in self._exam_results if res["patient_id"] == pid]
            return max(results, key=lambda item: item["created_at"], default=None)

        rows: List[Dict] = []
        for patient_id, data in self._patients.items():
            latest = latest_result(patient_id) or {}
            rows.append(
                {
                    **data,
                    "predicted_label": latest.get("predicted_label"),
                    "confidence": latest.get("confidence"),
                    "last_exam_at": latest.get("created_at"),
                }
            )
        rows.sort(key=lambda item: (item.get("last_name") or "", item.get("first_name") or ""))
        return rows

    def record_exam_result(
        self,
        patient_id: str,
        *,
        image_path: str,
        prediction: Dict,
        model_version: str,
    ) -> str:
        result_id = str(uuid.uuid4())
        record = {
            "result_id": result_id,
            "patient_id": patient_id,
            "image_path": image_path,
            "model_version": model_version,
            "predicted_label": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "probabilities": prediction.get("probabilities", {}),
            "created_at": datetime.utcnow(),
        }
        self._exam_results.append(record)
        return result_id

    def patient_history(self, patient_id: str, limit: int = 50) -> List[Dict]:
        history = [res for res in self._exam_results if res["patient_id"] == patient_id]
        history.sort(key=lambda item: item["created_at"], reverse=True)
        return history[:limit]

    def recent_results(self, limit: int = 25) -> List[Dict]:
        merged: List[Dict] = []
        for res in self._exam_results:
            patient = self._patients.get(res["patient_id"], {})
            merged.append(
                {
                    **res,
                    "first_name": patient.get("first_name"),
                    "last_name": patient.get("last_name"),
                }
            )
        merged.sort(key=lambda item: item["created_at"], reverse=True)
        return merged[:limit]

    def close(self) -> None:  # pragma: no cover - symmetry with DatabaseManager
        self._patients.clear()
        self._patients_by_national_id.clear()
        self._exam_results.clear()




class DatabaseManager:
    """Convenience wrapper around psycopg2 with small helper methods."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError(
                "DATABASE_URL is not set. Provide a Neon connection string or pass a DSN."
            )
        self._connection = self._connect()
        self._ensure_schema()

    def _connect(self):
        conn = psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn

    @contextmanager
    def cursor(self) -> Iterator[psycopg2.extensions.cursor]:
        cur = self._connection.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def _ensure_schema(self) -> None:
        with self.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            except psycopg2.Error:
                self._connection.rollback()
                # Neon already exposes gen_random_uuid; ignore permission errors.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    national_id VARCHAR(32) UNIQUE NOT NULL,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    birth_place VARCHAR(100),
                    birth_date DATE,
                    gender VARCHAR(16),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS exam_results (
                    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    patient_id UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                    image_path TEXT,
                    model_version VARCHAR(64),
                    predicted_label VARCHAR(100) NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    probabilities JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_exam_results_patient_id_created
                    ON exam_results (patient_id, created_at DESC);
                """
            )

    def upsert_patient(self, patient: PatientInfo) -> str:
        """Insert or update a patient, returning the patient UUID."""
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients (national_id, first_name, last_name, birth_place, birth_date, gender)
                VALUES (%(national_id)s, %(first_name)s, %(last_name)s, %(birth_place)s, %(birth_date)s, %(gender)s)
                ON CONFLICT (national_id) DO UPDATE
                SET first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    birth_place = EXCLUDED.birth_place,
                    birth_date = EXCLUDED.birth_date,
                    gender = EXCLUDED.gender
                RETURNING patient_id;
                """,
                asdict(patient),
            )
            row = cur.fetchone()
            return str(row["patient_id"])

    def list_patients(self) -> List[Dict]:
        """Return patients and their latest inference summary."""
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.patient_id,
                    p.national_id,
                    p.first_name,
                    p.last_name,
                    p.birth_place,
                    p.birth_date,
                    p.gender,
                    p.created_at,
                    latest.predicted_label,
                    latest.confidence,
                    latest.created_at AS last_exam_at
                FROM patients p
                LEFT JOIN LATERAL (
                    SELECT predicted_label, confidence, created_at
                    FROM exam_results er
                    WHERE er.patient_id = p.patient_id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) latest ON TRUE
                ORDER BY p.last_name, p.first_name;
                """
            )
            return list(cur.fetchall())

    def record_exam_result(
        self,
        patient_id: str,
        *,
        image_path: str,
        prediction: Dict,
        model_version: str,
    ) -> str:
        """Persist a single prediction and return the created result id."""
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO exam_results (patient_id, image_path, model_version, predicted_label, confidence, probabilities)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING result_id;
                """,
                (
                    patient_id,
                    image_path,
                    model_version,
                    prediction["label"],
                    prediction["confidence"],
                    json.dumps(prediction["probabilities"]),
                ),
            )
            row = cur.fetchone()
            return str(row["result_id"])

    def patient_history(self, patient_id: str, limit: int = 50) -> List[Dict]:
        """Fetch recent inference records for a patient."""
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT result_id,
                       image_path,
                       model_version,
                       predicted_label,
                       confidence,
                       probabilities,
                       created_at
                FROM exam_results
                WHERE patient_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (patient_id, limit),
            )
            rows = cur.fetchall()
            for row in rows:
                if isinstance(row.get("probabilities"), str):
                    row["probabilities"] = json.loads(row["probabilities"])
            return rows

    def recent_results(self, limit: int = 25) -> List[Dict]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT
                    er.result_id,
                    er.patient_id,
                    p.first_name,
                    p.last_name,
                    er.predicted_label,
                    er.confidence,
                    er.created_at
                FROM exam_results er
                JOIN patients p ON p.patient_id = er.patient_id
                ORDER BY er.created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def close(self) -> None:
        if self._connection:
            self._connection.close()






def create_database_manager(dsn: str | None = None) -> DatabaseBackend:
    """Return a concrete database backend, defaulting to in-memory storage."""
    resolved = dsn or os.getenv("DATABASE_URL")
    if resolved:
        return DatabaseManager(resolved)
    return InMemoryDatabase()


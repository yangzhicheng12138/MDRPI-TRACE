"""Opt-in learning records and user-rated FSRS reviews, separate from chat memory."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .preferences import validate_profile

RATING_LABELS = {
    1: ("还需再学", "Again"), 2: ("回忆困难", "Hard"),
    3: ("能够回忆", "Good"), 4: ("回忆轻松", "Easy"),
}
SCHEMA_VERSION = "learning_profile_v1"


def utc_now():
    return datetime.now(timezone.utc)


def question_key(record):
    item = record.get("bank_item") or {}
    # Keep changed questions/keys/rubrics as distinct review cards, including
    # old saved versions. Language changes alone do not create another card.
    value = {"id": item.get("id", ""), "case": item.get("case", record["case"]),
             "answer": item.get("answer"), "rubric": item.get("rubric")}
    flow = record.get("clinical_flow") or {}
    if flow.get("scoring_version") or record.get("scoring_version"):
        value["clinical_scoring_version"] = flow.get("scoring_version") or record["scoring_version"]
    review = (record.get("reviews") or [{}])[-1]
    if (review.get("objective_question_type") in {"true_false", "ordering"}
            and review.get("assessment_method") == "authored_objective_rules_v1"):
        value["objective_scoring_version"] = review["rubric_version"]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class LearningStore:
    def __init__(self, root, owner_id=""):
        self.path = Path(root) / "learning.sqlite"
        if not isinstance(owner_id, str) or owner_id and not re.fullmatch(r"[a-f0-9]{64}", owner_id):
            raise ValueError("学习档案访问标识无效。")
        self.owner_id = owner_id

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_key TEXT PRIMARY KEY, name TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS exercises (
                    record_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_key TEXT NOT NULL, generation INTEGER NOT NULL,
                    exercise_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    question_key TEXT NOT NULL, completed_at TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(profile_key, generation, exercise_id));
                CREATE TABLE IF NOT EXISTS cards (
                    profile_key TEXT NOT NULL, generation INTEGER NOT NULL,
                    question_key TEXT NOT NULL, card_json TEXT NOT NULL,
                    due TEXT NOT NULL, last_exercise_id TEXT NOT NULL,
                    PRIMARY KEY(profile_key, generation, question_key));
                CREATE TABLE IF NOT EXISTS ratings (
                    profile_key TEXT NOT NULL, generation INTEGER NOT NULL,
                    exercise_id TEXT NOT NULL, rating INTEGER NOT NULL,
                    reviewed_at TEXT NOT NULL, due TEXT NOT NULL, log_json TEXT NOT NULL,
                    PRIMARY KEY(profile_key, generation, exercise_id));
            """)
            # Serialize clears, record writes and scheduler updates across
            # sessions; callers use a fresh connection in each worker thread.
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _profile(self, connection, profile):
        name = validate_profile(profile)
        if not name:
            raise ValueError("请先在教学设置中填写学习档案名称。")
        # Empty owners retain the exact pre-existing namespace. Scoped names
        # never adopt, delete or expose a legacy same-name profile implicitly.
        identity = self.owner_id + ":" + name if self.owner_id else name
        key = hashlib.sha256(identity.encode()).hexdigest()
        connection.execute("INSERT OR IGNORE INTO profiles(profile_key,name) VALUES(?,?)", (key, name))
        return connection.execute("SELECT * FROM profiles WHERE profile_key=?", (key,)).fetchone()

    def generation(self, profile):
        with self._connection() as connection:
            return self._profile(connection, profile)["generation"]

    def save_completed(self, profile, generation, state):
        if not profile or state.get("mode") != "teaching" or state.get("stage") != "complete" or state.get("pending"):
            return {"saved": False, "reason": "not_completed_or_not_enabled"}
        return self._save(profile, generation, state)

    def save_progress(self, profile, generation, state):
        """Persist a genuine guided exercise after each settled learner turn."""
        flow = state.get("clinical_flow") or {}
        if (not profile or state.get("mode") != "teaching" or state.get("pending")
                or flow.get("status") not in {"in_progress", "interrupted", "completed"}
                or not state.get("case")):
            return {"saved": False, "reason": "not_guided_progress_or_not_enabled"}
        return self._save(profile, generation, state)

    @staticmethod
    def _completed(record):
        if record.get("clinical_flow"):
            from .learning_analytics import record_metrics
            return record_metrics(record)["completion_status"] == "completed"
        return record.get("stage") == "complete"

    def _save(self, profile, generation, state):
        from .learning import exercise_record
        record = exercise_record(state)
        # Preserve additive guided metadata even while an older deployed
        # exercise_record() helper is in use. No historical record is migrated.
        for field in ("clinical_flow", "case_spec", "case_metadata", "scoring_version"):
            if field in state:
                record[field] = deepcopy(state[field])
        encoded = json.dumps(record, ensure_ascii=False)
        with self._connection() as connection:
            owner = self._profile(connection, profile)
            if owner["generation"] != generation:
                return {"saved": False, "reason": "profile_was_cleared"}
            existing = connection.execute("SELECT completed_at,record_json FROM exercises WHERE profile_key=? AND generation=? AND exercise_id=?",
                (owner["profile_key"], generation, record["exercise_id"])).fetchone()
            saved_at = utc_now().isoformat()
            if existing and (self._completed(json.loads(existing["record_json"])) or not self._completed(record)):
                saved_at = existing["completed_at"]
            connection.execute("""INSERT INTO exercises
                (profile_key,generation,exercise_id,session_id,question_key,completed_at,record_json)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(profile_key,generation,exercise_id)
                DO UPDATE SET record_json=excluded.record_json,completed_at=excluded.completed_at""",
                (owner["profile_key"], generation, record["exercise_id"], state["session_id"],
                 question_key(record), saved_at, encoded))
        return {"saved": True, "profile": profile, "exercise_id": record["exercise_id"]}

    def _rows(self, connection, owner):
        return connection.execute("""SELECT e.*, c.due AS next_due, r.rating,
            r.reviewed_at AS rated_at FROM exercises e
            LEFT JOIN cards c ON c.profile_key=e.profile_key AND c.generation=e.generation
                AND c.question_key=e.question_key
            LEFT JOIN ratings r ON r.profile_key=e.profile_key AND r.generation=e.generation
                AND r.exercise_id=e.exercise_id
            WHERE e.profile_key=? AND e.generation=? ORDER BY e.record_order""",
            (owner["profile_key"], owner["generation"])).fetchall()

    def progress(self, profile, *, at=None):
        from .learning import row_for
        from .learning_analytics import aggregate_records, record_metrics
        at = at or utc_now()
        with self._connection() as connection:
            owner = self._profile(connection, profile)
            rows = self._rows(connection, owner)
            result, records = [], []
            for row in rows:
                record = json.loads(row["record_json"])
                records.append(record)
                data = row_for(record, record.get("language", "zh"))
                complete = self._completed(record)
                if record.get("clinical_flow"):
                    metrics = record_metrics(record)
                    data.update(complete=complete, completion_status=metrics["completion_status"],
                                passed_steps=metrics["passed_steps"], hint_count=metrics["hint_count"])
                    if not complete:
                        data["outcome"] = "interrupted" if metrics["completion_status"] == "interrupted" else "in_progress"
                data.update(source_session_id=row["session_id"], question_key=row["question_key"],
                            completed_at=row["completed_at"] if complete else "", next_due=row["next_due"] if complete else None,
                            self_rating=row["rating"], self_rated_at=row["rated_at"])
                result.append(data)
            latest = {row["question_key"]: row for row in result}
            questions = list(latest.values())
            completed = {row["question_key"]: row for row in result if row["complete"]}
            due = [row for row in completed.values() if row["next_due"] and datetime.fromisoformat(row["next_due"]) <= at]
            return {"schema_version": SCHEMA_VERSION, "profile": owner["name"],
                    "generation": owner["generation"], "completed_rounds": sum(row["complete"] for row in result),
                    "unique_questions": len(questions), "due_count": len(due),
                    "unrated_count": sum(row["self_rating"] is None for row in completed.values()),
                    "due_questions": sorted(due, key=lambda row: row["next_due"]),
                    "questions": questions, "exercises": result,
                    "analytics": aggregate_records(records),
                    "schedule_basis": "learner_self_rating_fsrs_6.3.2"}

    def raw_records(self, profile):
        """Return only this owner's current-generation original exercise data."""
        with self._connection() as connection:
            owner = self._profile(connection, profile)
            return [json.loads(row["record_json"]) for row in self._rows(connection, owner)]

    def rate(self, profile, generation, exercise_id, rating, *, at=None):
        from fsrs import Scheduler, Card, Rating
        if isinstance(rating, bool) or rating not in RATING_LABELS:
            raise ValueError("复习自评需为1、2、3或4。")
        at = at or utc_now()
        with self._connection() as connection:
            owner = self._profile(connection, profile)
            if owner["generation"] != generation:
                raise ValueError("学习档案已清空，请完成一次新的练习后自评。")
            identity = (owner["profile_key"], generation, exercise_id)
            record = connection.execute("SELECT * FROM exercises WHERE profile_key=? AND generation=? AND exercise_id=?", identity).fetchone()
            if record is None:
                raise ValueError("本档案中没有这次已完成的练习。")
            if not self._completed(json.loads(record["record_json"])):
                raise ValueError("请完成六步训练后再进行复习自评；进行中或中断的训练不会生成复习安排。")
            rated = connection.execute("SELECT * FROM ratings WHERE profile_key=? AND generation=? AND exercise_id=?", identity).fetchone()
            if rated:
                if rated["rating"] != rating:
                    raise ValueError("这轮练习已经自评；再次练习后可以提交新的自评。")
                return {"exercise_id": exercise_id, "rating": rating, "next_due": rated["due"], "already_rated": True}
            newer = connection.execute("SELECT record_json FROM exercises WHERE profile_key=? AND generation=? AND question_key=? AND record_order>?",
                (owner["profile_key"], generation, record["question_key"], record["record_order"])).fetchall()
            if any(self._completed(json.loads(row["record_json"])) for row in newer):
                raise ValueError("这道题已有更新的完成记录，请为最新一轮自评。")
            card_key = (owner["profile_key"], generation, record["question_key"])
            existing = connection.execute("SELECT card_json FROM cards WHERE profile_key=? AND generation=? AND question_key=?", card_key).fetchone()
            card = Card.from_json(existing["card_json"]) if existing else Card()
            scheduler = Scheduler(enable_fuzzing=False, maximum_interval=365)
            card, review_log = scheduler.review_card(card, Rating(rating), review_datetime=at)
            connection.execute("INSERT INTO cards VALUES(?,?,?,?,?,?) ON CONFLICT(profile_key,generation,question_key) DO UPDATE SET card_json=excluded.card_json,due=excluded.due,last_exercise_id=excluded.last_exercise_id",
                (*card_key, card.to_json(), card.due.isoformat(), exercise_id))
            connection.execute("INSERT INTO ratings VALUES(?,?,?,?,?,?,?)", (*identity, rating, at.isoformat(), card.due.isoformat(), review_log.to_json()))
            return {"exercise_id": exercise_id, "rating": rating, "next_due": card.due.isoformat(), "already_rated": False}

    def due_record(self, profile, *, at=None):
        at = at or utc_now()
        with self._connection() as connection:
            owner = self._profile(connection, profile)
            rows = self._rows(connection, owner)
            latest = {row["question_key"]: row for row in rows if self._completed(json.loads(row["record_json"]))}
            due = [row for row in latest.values() if row["next_due"] and datetime.fromisoformat(row["next_due"]) <= at]
            if not due:
                raise ValueError("目前没有到期复习题，可以先选择新练习。")
            row = min(due, key=lambda value: value["next_due"])
            record = json.loads(row["record_json"])
            record.update(source_session_id=row["session_id"], profile_generation=owner["generation"])
            return record

    def export(self, profile):
        with self._connection() as connection:
            owner = self._profile(connection, profile)
            key = (owner["profile_key"], owner["generation"])
            return {"schema_version": SCHEMA_VERSION, "profile": owner["name"], "exported_at": utc_now().isoformat(),
                    "records": [{"source_session_id": row["session_id"], "completed_at": row["completed_at"],
                                 "exercise": json.loads(row["record_json"])} for row in self._rows(connection, owner)],
                    "review_logs": [json.loads(row["log_json"]) for row in connection.execute(
                        "SELECT log_json FROM ratings WHERE profile_key=? AND generation=? ORDER BY reviewed_at", key)],
                    "scheduler": {"library": "fsrs", "version": "6.3.2", "fuzzing": False, "maximum_interval_days": 365}}

    def clear(self, profile):
        with self._connection() as connection:
            owner = self._profile(connection, profile)
            for table in ("exercises", "ratings", "cards"):
                connection.execute(f"DELETE FROM {table} WHERE profile_key=?", (owner["profile_key"],))
            connection.execute("UPDATE profiles SET generation=generation+1 WHERE profile_key=?", (owner["profile_key"],))
            return {"profile": owner["name"], "generation": owner["generation"] + 1, "cleared": True}

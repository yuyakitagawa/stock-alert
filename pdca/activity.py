#!/usr/bin/env python3
"""アクティビティログ: 各担当者（FM/Quant/Securities/Engineer/System）が
   実施中・実施済みのアクションを Supabase の activity_log に記録する。
   そこを見れば誰でも「今チームが何をしているか / 何をしたか」を把握できる。

   使い方:
     aid = start("Quant", "パラメータ改善提案", "検討中…")   # 実施中（running）を記録
     finish(aid, "done", "n_estimators 5000→8000", 詳細全文)  # 完了時に更新
     record("FM", "改善方針の判断", "improve", 理由)           # 一発記録
"""
import os, requests
from pathlib import Path
from datetime import date, datetime, timezone
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SVC = os.getenv("SUPABASE_SERVICE_KEY", "")
TABLE        = "activity_log"


def _enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SVC)


def _headers(extra: dict | None = None) -> dict:
    h = {
        "apikey":        SUPABASE_SVC,
        "Authorization": f"Bearer {SUPABASE_SVC}",
        "Content-Type":  "application/json",
    }
    if extra:
        h.update(extra)
    return h


def start(role: str, step: str, summary: str = "", detail: str = "") -> int | None:
    """実施中（running）を記録し、行 id を返す（失敗時 None）。"""
    print(f"  [activity] {role}: {step} … 実施中")
    if not _enabled():
        return None
    row = {
        "run_date": date.today().isoformat(),
        "role":     role,
        "step":     step,
        "status":   "running",
        "summary":  summary,
        "detail":   detail,
    }
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_headers({"Prefer": "return=representation"}),
            json=[row], timeout=15,
        )
        if r.ok and r.json():
            return r.json()[0]["id"]
        print(f"  [activity start error] {r.status_code} {r.text[:150]}")
    except Exception as e:
        print(f"  [activity start error] {e}")
    return None


def finish(aid: int | None, status: str = "done",
           summary: str | None = None, detail: str | None = None) -> None:
    """start() で作った行を完了状態に更新する。"""
    print(f"  [activity] → {status}")
    if not _enabled() or aid is None:
        return
    patch = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if summary is not None:
        patch["summary"] = summary
    if detail is not None:
        patch["detail"] = detail
    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/{TABLE}?id=eq.{aid}",
            headers=_headers(), json=patch, timeout=15,
        )
    except Exception as e:
        print(f"  [activity finish error] {e}")


def record(role: str, step: str, status: str = "done",
           summary: str = "", detail: str = "") -> None:
    """running を挟まず一発で記録する（瞬時に終わるアクション用）。"""
    print(f"  [activity] {role}: {step} ({status})")
    if not _enabled():
        return
    row = {
        "run_date": date.today().isoformat(),
        "role":     role,
        "step":     step,
        "status":   status,
        "summary":  summary,
        "detail":   detail,
    }
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_headers(), json=[row], timeout=15,
        )
    except Exception as e:
        print(f"  [activity record error] {e}")

"""yahoo_price_cache のローカルミラー。

なぜ要るか:
  rank_stocks / rf_train_v3 / backtest は「全銘柄 × 400〜600日」の終値を毎回読む。
  これを PostgREST 経由でやると1銘柄1リクエスト・合計100万行超で、1回あたり約52MBの
  egressになっていた（平日毎日 + 金曜の再学習 + backtestの手動実行）。Supabase Free枠の
  egress 5GB/月はこれだけでほぼ埋まり、2026-09-07に402で全RESTが止まった。

やり方:
  全履歴を1度だけpickleに落とし、以降は `date > 手元の最終日` の差分だけを
  1リクエスト（ページング込み）で取る。1日あたり約3,800行 = 約0.2MB。
  GitHub Actions では actions/cache に載せてrun間で持ち回す（daily_alert.yml）。
  手元では BASE_DIR に残るのでbacktestの再実行がタダになる。

形式:
  {code: (dates np.datetime64[D], closes float64, volumes float64)} を pickle 化。
  pyarrow を足したくないので parquet ではなく pickle。Actions cache 側で圧縮される。
"""
import os
import pickle
import threading
from datetime import date, timedelta

import numpy as np

import lib.supabase_client as sb

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.getenv("STOCK_ALERT_HOME", _PROJECT_DIR)
STORE_PATH = os.path.join(BASE_DIR, "_price_store.pkl")

# 手元のミラーとSupabaseがずれる経路が2つある: (1) data_sanity 等による過去行の訂正、
# (2) 差分取得より前に入った古い日付の行。どちらも差分同期では拾えないので、
# この日数を超えたら全件を取り直す。
_FULL_REBUILD_AFTER_DAYS = 30

_lock = threading.Lock()
_store: dict | None = None


def _empty() -> dict:
    return {"built_at": None, "max_date": None, "prices": {}}


def _read_disk() -> dict:
    if not os.path.exists(STORE_PATH):
        return _empty()
    try:
        with open(STORE_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "prices" in data:
            return data
    except Exception as e:
        print(f"[price_store] 読み込み失敗（作り直します）: {e}")
    return _empty()


def _write_disk(store: dict) -> None:
    tmp = STORE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(store, f, protocol=5)
    os.replace(tmp, STORE_PATH)


def _rows_to_prices(rows: list[dict]) -> dict:
    """REST行を {code: (dates, closes, volumes)} にまとめる。"""
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r["code"], []).append((r["date"], r["close"], r["volume"]))
    out = {}
    for code, vals in buckets.items():
        vals.sort(key=lambda v: v[0])
        out[code] = (
            np.array([v[0] for v in vals], dtype="datetime64[D]"),
            np.array([np.nan if v[1] is None else v[1] for v in vals], dtype="float64"),
            np.array([np.nan if v[2] is None else v[2] for v in vals], dtype="float64"),
        )
    return out


def _merge(existing: dict, incoming: dict) -> dict:
    """銘柄ごとに日付でマージする。同じ日付は incoming（Supabase側）を採る。"""
    for code, (d2, c2, v2) in incoming.items():
        if code not in existing:
            existing[code] = (d2, c2, v2)
            continue
        d1, c1, v1 = existing[code]
        keep = ~np.isin(d1, d2)
        dates = np.concatenate([d1[keep], d2])
        order = np.argsort(dates, kind="stable")
        existing[code] = (
            dates[order],
            np.concatenate([c1[keep], c2])[order],
            np.concatenate([v1[keep], v2])[order],
        )
    return existing


def _fetch(query: str) -> list[dict]:
    """全ページ揃わなければ例外。取れた分だけで先へ進むとミラーに穴が空き、
    以後の差分同期では二度と埋まらない。"""
    return sb.select("yahoo_price_cache", query, strict=True)


# 全履歴を一度に引くと offset が深くなり、35万行あたりで statement timeout になる
# （実測）。主キー(code,date)の順に銘柄を小分けして引けば offset は常に浅い。
_FETCH_CHUNK_CODES = 40


def _fetch_codes(codes: list[str]) -> dict:
    prices: dict = {}
    total = 0
    for i in range(0, len(codes), _FETCH_CHUNK_CODES):
        chunk = codes[i: i + _FETCH_CHUNK_CODES]
        codes_q = ",".join(f'"{c}"' for c in chunk)
        rows = _fetch(f"code=in.({codes_q})&select=code,date,close,volume"
                      "&order=code.asc,date.asc")
        total += len(rows)
        prices = _merge(prices, _rows_to_prices(rows))
        if (i // _FETCH_CHUNK_CODES) % 20 == 0:
            print(f"[price_store]   {i + len(chunk)}/{len(codes)}銘柄 ({total:,}行)")
    print(f"[price_store] {total:,}行 取得")
    return prices


def _full_fetch() -> dict:
    codes = rpc_codes()
    print(f"[price_store] 全履歴を取得します（初回のみ・{len(codes)}銘柄）")
    return _fetch_codes(codes)


def _sync(store: dict) -> dict:
    """差分だけ取り込む。手元に無い銘柄は全履歴を取る。"""
    max_date = store.get("max_date")
    built_at = store.get("built_at")
    stale = (
        built_at is None
        or (date.today() - date.fromisoformat(built_at)).days >= _FULL_REBUILD_AFTER_DAYS
    )
    if not store["prices"] or max_date is None or stale:
        store["prices"] = _full_fetch()
        store["built_at"] = date.today().isoformat()
    else:
        rows = _fetch(f"date=gt.{max_date}&select=code,date,close,volume&order=date.asc")
        if rows:
            print(f"[price_store] 差分 {len(rows):,}行（{max_date} より後）")
            store["prices"] = _merge(store["prices"], _rows_to_prices(rows))

        # fetch_history が新規上場銘柄を過去分ごと入れた場合、差分同期では
        # 当日の1行しか入らず「行数不足」で毎日落ち続ける。コード一覧だけ突き合わせる。
        remote = rpc_codes()
        missing = [c for c in remote if c not in store["prices"]] if remote else []
        if missing:
            print(f"[price_store] 手元に無い銘柄 {len(missing)}件の履歴を取得します")
            store["prices"] = _merge(store["prices"], _fetch_codes(missing))

    latest = [d[-1] for d, _, _ in store["prices"].values() if len(d)]
    store["max_date"] = str(max(latest)) if latest else None
    _write_disk(store)
    return store


def rpc_codes() -> list[str]:
    """Supabase側の銘柄コード一覧（RPC。161万行のcode列を引かない）。

    setof を返すRPCは PostgREST の max-rows(1000) で切られるため、DB側で
    text[] に畳んで1行で返している。
    """
    codes = sb.rpc("price_cache_codes", {})
    if not isinstance(codes, list):
        print("[price_store] 銘柄一覧の取得に失敗（新規上場銘柄の取り込みを見送ります）")
        return []
    return list(codes)


_enabled = False


def enable() -> None:
    """このプロセスでミラーを使う（構築・差分同期はここで走る）。

    全銘柄を舐める処理（rank_stocks / rf_train_v3 / backtest / fetch_history）だけが
    呼ぶこと。1銘柄だけ引く処理（毎時のブログ生成など）が呼ぶと、たかだか数百行の
    ために全履歴の構築が走って逆効果になる。
    """
    global _enabled
    _enabled = True
    _get()


def is_enabled() -> bool:
    return _enabled


def _get() -> dict:
    global _store
    if _store is not None:
        return _store
    with _lock:
        if _store is None:
            _store = _sync(_read_disk())
    return _store


# ── 参照API ────────────────────────────────────────────────────────────────

def codes() -> list[str]:
    return sorted(_get()["prices"].keys())


def coverage(code: str):
    """(min_date_str, max_date_str)。未収録は None。"""
    entry = _get()["prices"].get(str(code))
    if entry is None or len(entry[0]) == 0:
        return None
    return str(entry[0][0]), str(entry[0][-1])


def frame(code: str, start: str | None = None, end: str | None = None, days: int | None = None):
    """DataFrame(Close, Volume) を返す。index は datetime.date。該当なしは None。"""
    import pandas as pd

    entry = _get()["prices"].get(str(code))
    if entry is None:
        return None
    dates, closes, vols = entry
    mask = np.ones(len(dates), dtype=bool)
    if days:
        mask &= dates >= np.datetime64(date.today() - timedelta(days=days))
    if start:
        mask &= dates >= np.datetime64(start)
    if end:
        mask &= dates <= np.datetime64(end)
    if not mask.any():
        return None
    return pd.DataFrame(
        {"Close": closes[mask], "Volume": vols[mask]},
        index=[d.astype(object) for d in dates[mask]],
    )


def absorb(code: str, rows: list[dict]) -> None:
    """Supabaseへ書いた行を手元にも反映する（同一プロセス内の読み直しを防ぐ）。"""
    if not _enabled or _store is None or not rows:
        return
    with _lock:
        _store["prices"] = _merge(_store["prices"], _rows_to_prices(rows))

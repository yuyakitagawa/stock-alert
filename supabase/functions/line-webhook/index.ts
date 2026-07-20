import Anthropic from "https://esm.sh/@anthropic-ai/sdk@0.39.0";

const LINE_CHANNEL_SECRET = Deno.env.get("LINE_CHANNEL_SECRET") ?? "";
const LINE_CHANNEL_ACCESS_TOKEN = Deno.env.get("LINE_CHANNEL_ACCESS_TOKEN") ?? "";
const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY") ?? "";

const SB_URL = (Deno.env.get("SUPABASE_URL") ?? "").trim();
const SB_KEY = (Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "").trim();

const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = 5;
const rateLimitMap = new Map<string, number[]>();

const NOTE_AUTH_CODE = Deno.env.get("NOTE_AUTH_CODE") ?? "";

// ── ユーザー管理 ─────────────────────────────

interface UserRecord {
  line_user_id: string;
  plan: "free" | "premium";
  expires_at: string | null;
  total_ai_queries: number;
}

async function getOrCreateUser(userId: string): Promise<UserRecord> {
  const res = await fetch(
    `${SB_URL}/rest/v1/line_users?line_user_id=eq.${userId}&select=line_user_id,plan,expires_at,total_ai_queries`,
    { headers: sbHeaders() },
  );
  const rows = res.ok ? await res.json() : [];
  if (rows.length > 0) {
    return rows[0] as UserRecord;
  }
  await fetch(`${SB_URL}/rest/v1/line_users`, {
    method: "POST",
    headers: sbHeaders({ Prefer: "return=minimal" }),
    body: JSON.stringify({ line_user_id: userId }),
  });
  return { line_user_id: userId, plan: "free", expires_at: null, total_ai_queries: 0 };
}

function isPremium(user: UserRecord): boolean {
  return user.plan === "premium";
}

const FREE_AI_LIMIT = 3;
const FREE_WATCHLIST_LIMIT = 3;
const FREE_RANKING_LIMIT = 5;
const dailyAiUsage = new Map<string, { date: string; count: number }>();

function checkAiLimit(userId: string): { allowed: boolean; remaining: number } {
  const today = new Date().toISOString().slice(0, 10);
  const usage = dailyAiUsage.get(userId);
  if (!usage || usage.date !== today) {
    dailyAiUsage.set(userId, { date: today, count: 0 });
    return { allowed: true, remaining: FREE_AI_LIMIT };
  }
  return { allowed: usage.count < FREE_AI_LIMIT, remaining: FREE_AI_LIMIT - usage.count };
}

const UPGRADE_NUDGE_INTERVAL = 10;
const UPGRADE_NUDGE = "\n\n💡 プレミアムプランならAI相談が無制限！noteサブスクで「認証 コード」を送信するとアップグレードできます。";

async function incrementTotalAiQueries(userId: string): Promise<number> {
  const res = await fetch(
    `${SB_URL}/rest/v1/rpc/increment_total_ai_queries`,
    {
      method: "POST",
      headers: sbHeaders(),
      body: JSON.stringify({ p_user_id: userId }),
    },
  );
  if (res.ok) {
    const val = await res.json();
    return typeof val === "number" ? val : 0;
  }
  return 0;
}

function recordAiUsage(userId: string): void {
  const today = new Date().toISOString().slice(0, 10);
  const usage = dailyAiUsage.get(userId);
  if (!usage || usage.date !== today) {
    dailyAiUsage.set(userId, { date: today, count: 1 });
  } else {
    usage.count++;
  }
}

function sbHeaders(extra: Record<string, string> = {}) {
  return {
    apikey: SB_KEY,
    Authorization: `Bearer ${SB_KEY}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

async function verifySignature(body: string, signature: string): Promise<boolean> {
  if (!LINE_CHANNEL_SECRET) return false;
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(LINE_CHANNEL_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(body));
  const hash = btoa(String.fromCharCode(...new Uint8Array(sig)));
  return hash === signature;
}

function checkRateLimit(userId: string): boolean {
  const now = Date.now();
  const timestamps = rateLimitMap.get(userId) ?? [];
  const recent = timestamps.filter((t) => now - t < RATE_LIMIT_WINDOW_MS);
  if (recent.length >= RATE_LIMIT_MAX) return false;
  recent.push(now);
  rateLimitMap.set(userId, recent);
  return true;
}

async function replyToLine(replyToken: string, text: string): Promise<void> {
  const maxLen = 5000;
  const truncated = text.length > maxLen ? text.slice(0, maxLen - 3) + "..." : text;
  await fetch("https://api.line.me/v2/bot/message/reply", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LINE_CHANNEL_ACCESS_TOKEN}`,
    },
    body: JSON.stringify({
      replyToken,
      messages: [{ type: "text", text: truncated }],
    }),
  });
}

async function showLoadingAnimation(userId: string, seconds = 60): Promise<void> {
  await fetch("https://api.line.me/v2/bot/chat/loading/start", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LINE_CHANNEL_ACCESS_TOKEN}`,
    },
    body: JSON.stringify({ chatId: userId, loadingSeconds: seconds }),
  });
}

async function pushToLine(userId: string, text: string): Promise<void> {
  const maxLen = 5000;
  const truncated = text.length > maxLen ? text.slice(0, maxLen - 3) + "..." : text;
  await fetch("https://api.line.me/v2/bot/message/push", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LINE_CHANNEL_ACCESS_TOKEN}`,
    },
    body: JSON.stringify({
      to: userId,
      messages: [{ type: "text", text: truncated }],
    }),
  });
}

// ── ウォッチリスト管理 ────────────────────────

interface WatchItem {
  code: string;
  name: string;
  dp_threshold: number;
  dp_sell_threshold: number;
  line_user_id: string;
}

async function getWatchlist(userId: string): Promise<WatchItem[]> {
  const res = await fetch(
    `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold,dp_sell_threshold&order=created_at`,
    { headers: sbHeaders() },
  );
  return res.ok ? await res.json() : [];
}

async function addToWatchlist(
  userId: string,
  code: string,
  name: string,
  threshold: number,
  sellThreshold: number = 20.0,
): Promise<boolean> {
  const res = await fetch(`${SB_URL}/rest/v1/dp_watchlist`, {
    method: "POST",
    headers: sbHeaders({ Prefer: "resolution=merge-duplicates" }),
    body: JSON.stringify({
      line_user_id: userId,
      code,
      name,
      dp_threshold: threshold,
      dp_sell_threshold: sellThreshold,
    }),
  });
  return res.ok;
}

async function removeFromWatchlist(userId: string, code: string): Promise<boolean> {
  const res = await fetch(
    `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&code=eq.${code}`,
    { method: "DELETE", headers: sbHeaders() },
  );
  return res.ok;
}

async function lookupStock(
  code: string,
): Promise<{ code: string; name: string; close: number; drop_prob: number } | null> {
  const res = await fetch(
    `${SB_URL}/rest/v1/gen_rankings?code=eq.${code}&select=code,name,close,drop_prob&order=date.desc&limit=1`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return rows.length > 0 ? rows[0] : null;
}

async function fetchDropProbHistory(code: string, days = 30): Promise<string> {
  const res = await fetch(
    `${SB_URL}/rest/v1/gen_rankings?code=eq.${code}&select=date,name,close,drop_prob,net&order=date.desc&limit=${days}`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "";
  const rows: { date: string; name: string; close: number; drop_prob: number; net: number }[] =
    await res.json();
  if (rows.length === 0) return "データが見つかりません。";

  const name = rows[0].name;
  const sorted = [...rows].reverse(); // 古い順
  const lines = [`【${name}(${code}) 下落確率の推移】`];
  for (const r of sorted) {
    const d = r.date ? r.date.slice(5) : ""; // MM-DD
    const mark = r.drop_prob >= 20 ? "🔴" : r.drop_prob >= 12 ? "🟡" : "🟢";
    lines.push(`${d} ${mark}${r.drop_prob}% 株価:${r.close}円`);
  }
  return lines.join("\n");
}

function toFullWidth(s: string): string {
  return s.replace(/[A-Za-z0-9]/g, (c) => String.fromCharCode(c.charCodeAt(0) + 0xfee0));
}

// ── 株価の前日比・前週比 ──────────────────────────────────────────────

// 前週比のlookback日数（配列は直近何件保持していても、最大でここまで遡る）
const WEEK_LOOKBACK_DAYS = 5;

function pctChange(from: number, to: number): string | null {
  if (!from) return null; // 0/undefinedによるInfinity・NaN表示を防ぐ
  const pct = ((to - from) / from) * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

// rowsはdate.desc（[0]=最新）。前日比・前週比の文字列断片（例: "前日比+1.2%"）を返す。
// 保有履歴が浅く前週比が前日比と同じ行を指してしまう場合（rows.length<=2）は前週比を省略する。
function computePriceChanges(rows: { close: number }[]): string[] {
  if (rows.length < 2) return [];
  const parts: string[] = [];
  const day = pctChange(rows[1].close, rows[0].close);
  if (day) parts.push(`前日比${day}`);
  const weekIdx = Math.min(WEEK_LOOKBACK_DAYS, rows.length - 1);
  if (weekIdx > 1) {
    const week = pctChange(rows[weekIdx].close, rows[0].close);
    if (week) parts.push(`前週比${week}`);
  }
  return parts;
}

// ── EDINET大量保有: 売却/自己申告/個人名の判定 ──
// is_sell_disclosure/is_individual_filerはtools/scan_large_holdings.pyと同じキーワード判定。
// ただし「自己申告のみ除外し、売却は除外せず方向性(📉/📈)を表示する」方針は
// web/market_timing_alert.py の get_recent_large_holdings/build_large_holdings_section
// （PR #159）を踏襲している。scan_large_holdings.py の is_noise_match は
// self_filing/sell の両方をノイズとして除外する点が異なるので注意。

const SELL_KEYWORDS = ["譲渡", "売却", "売出", "処分"];
// これ以上は株式併合等によるスクイーズアウト（完全子会社化）の対象になりうる水準で、
// 上値が買取価格に収斂し伸びしろが無いとみなして除外する
const MAJORITY_HOLDING_THRESHOLD = 51;
const INSTITUTION_KEYWORDS = [
  "株式会社", "有限会社", "合同会社", "合資会社", "合名会社",
  "ホールディングス", "Holdings", "HD",
  "ファンド", "Fund", "キャピタル", "Capital",
  "パートナーズ", "Partners", "Investment", "投資顧問", "投資法人", "投資事業組合",
  "アセットマネジメント", "Asset Management", "証券", "信託銀行", "銀行",
  "生命保険", "損害保険", "Inc", "LLC", "LLP", "Ltd", "Corporation", "Corp",
];

function normalizeCompanyName(s: string): string {
  let out = s || "";
  for (const tok of ["株式会社", "(株)", "（株）", "ホールディングス", "HD", " ", "　", "・"]) {
    out = out.split(tok).join("");
  }
  return out.trim();
}

function isSellDisclosure(docDescription: string): boolean {
  return SELL_KEYWORDS.some((k) => (docDescription || "").includes(k));
}

function isSelfFiling(filerName: string, issuerName: string): boolean {
  const f = normalizeCompanyName(filerName);
  const i = normalizeCompanyName(issuerName);
  return !!(f && i && (f.includes(i) || i.includes(f)));
}

function isIndividualFiler(filerName: string): boolean {
  const normalized = (filerName || "").normalize("NFKC").trim();
  if (!normalized) return false;
  const lower = normalized.toLowerCase();
  if (INSTITUTION_KEYWORDS.some((kw) => lower.includes(kw.toLowerCase()))) return false;
  return normalized.includes(" ");
}

async function searchStockByName(keyword: string): Promise<{ code: string; name: string }[]> {
  const queries = [keyword];
  const fw = toFullWidth(keyword);
  if (fw !== keyword) queries.push(fw);

  for (const q of queries) {
    const res = await fetch(
      `${SB_URL}/rest/v1/jpx_stock_list?name=ilike.*${encodeURIComponent(q)}*&select=code,name&limit=5`,
      { headers: sbHeaders() },
    );
    const rows = res.ok ? await res.json() : [];
    if (rows.length > 0) return rows;
  }
  return [];
}

// ── コマンド処理 ─────────────────────────────

type CommandResult = { handled: true; reply: string } | { handled: false };

async function handleCommand(text: string, userId: string, user?: UserRecord): Promise<CommandResult> {
  const trimmed = text.trim();

  const authMatch = trimmed.match(/^(認証|auth)\s+(.+)$/i);
  if (authMatch) {
    const code = authMatch[2].trim();
    if (!NOTE_AUTH_CODE) {
      return { handled: true, reply: "認証機能は現在準備中です。" };
    }
    if (code !== NOTE_AUTH_CODE) {
      return { handled: true, reply: "❌ 認証コードが正しくありません。noteサブスクの限定記事をご確認ください。" };
    }
    await fetch(`${SB_URL}/rest/v1/line_users?line_user_id=eq.${userId}`, {
      method: "PATCH",
      headers: sbHeaders(),
      body: JSON.stringify({
        plan: "premium",
        auth_code: code,
        updated_at: new Date().toISOString(),
      }),
    });
    return {
      handled: true,
      reply: `🎉 プレミアムプランが有効になりました！\n\nAI相談無制限・ウォッチリスト無制限・ランキング詳細が利用可能です。`,
    };
  }

  if (/^(プラン|plan|状態|ステータス)$/i.test(trimmed)) {
    if (user && isPremium(user)) {
      return { handled: true, reply: `⭐ プレミアムプラン\n\nAI相談無制限 / ウォッチリスト無制限 / ランキング詳細` };
    }
    const aiStatus = checkAiLimit(userId);
    return { handled: true, reply: `📋 無料プラン\n\nAI相談: 残り${aiStatus.remaining}回/日\nウォッチリスト: 上限${FREE_WATCHLIST_LIMIT}銘柄\nランキング: 上位${FREE_RANKING_LIMIT}銘柄\n\nnoteサブスクで「認証 コード」を送信するとプレミアムにアップグレードできます。` };
  }

  if (/^(ヘルプ|help|使い方|何ができる|機能)$/i.test(trimmed)) {
    return {
      handled: true,
      reply:
        "📖 stock-alert Bot の使い方\n\n" +
        "【銘柄を調べる】\n" +
        "  4桁コードを送信 → 株価・下落確率を表示\n" +
        "  例: 8473\n\n" +
        "【ウォッチリスト】\n" +
        "  「ウォッチ 8473」→ 追加\n" +
        "  「解除 8473」→ 削除\n" +
        "  「リスト」→ 一覧表示\n\n" +
        "【ランキング】\n" +
        "  「ランキング」→ 本日のランキング\n\n" +
        "【AI相談】\n" +
        "  上記以外のメッセージ → AIが株の相談に回答\n" +
        "  例:「トヨタって今買い？」「高配当でおすすめは？」\n\n" +
        "【プラン管理】\n" +
        "  「プラン」→ 現在のプラン確認\n" +
        "  「認証 コード」→ プレミアム認証\n\n" +
        "【条件スクリーニング】\n" +
        "  「高配当で安全な銘柄は？」→ 条件検索\n" +
        "  「PBR1倍割れの割安株」→ 条件検索\n\n" +
        "【決算サプライズ速報】\n" +
        "  「最近の上方修正は？」→ カタリスト検索\n" +
        "  「ウォッチ銘柄のニュース」→ WL限定検索\n\n" +
        "【自動通知】\n" +
        "  毎日20時に日経シグナル＋ウォッチリスト状態をお届け",
    };
  }

  if (/^(ランキング|トップ|top|順位)$/i.test(trimmed)) {
    const premium = user ? isPremium(user) : false;
    const limit = premium ? 10 : FREE_RANKING_LIMIT;
    const today = new Date().toISOString().slice(0, 10);
    const res = await fetch(
      `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob&order=drop_prob.asc&limit=${limit}`,
      { headers: sbHeaders() },
    );
    if (!res.ok) return { handled: true, reply: "ランキングデータの取得に失敗しました。" };
    const rows = await res.json();
    if (rows.length === 0) {
      return { handled: true, reply: `${today} のランキングデータはまだありません。20時以降に更新されます。` };
    }
    const lines = [`📊 本日のトップ${limit}（${today}）\n`];
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const dpStatus = r.drop_prob < 8 ? "🟢" : r.drop_prob >= 15 ? "🔴" : "🟡";
      lines.push(`${i + 1}. ${r.code} ${r.name}\n   ${r.close.toLocaleString()}円 下落確率${r.drop_prob}% ${dpStatus}`);
    }
    if (!premium) {
      lines.push(`\n※ プレミアムプランでトップ10表示。「プラン」で詳細確認`);
    }
    return { handled: true, reply: lines.join("\n") };
  }

  if (/^(市場|相場|日経|マーケット)$/i.test(trimmed) || /今日の?状況|本日の?状況|今日どう|相場どう/.test(trimmed)) {
    const [n225Res, latestDateRes] = await Promise.all([
      fetch(
        `${SB_URL}/rest/v1/yahoo_market_index?ticker=eq.N225&order=date.desc&limit=1&select=date,close`,
        { headers: sbHeaders() },
      ),
      fetch(
        `${SB_URL}/rest/v1/gen_rankings?select=date&order=date.desc&limit=1`,
        { headers: sbHeaders() },
      ),
    ]);
    const n225 = n225Res.ok ? await n225Res.json() : [];
    const latestRows = latestDateRes.ok ? await latestDateRes.json() : [];
    const latestDate = latestRows.length > 0 ? latestRows[0].date : null;

    const rankRes = latestDate
      ? await fetch(
          `${SB_URL}/rest/v1/gen_rankings?date=eq.${latestDate}&select=drop_prob&limit=500`,
          { headers: sbHeaders() },
        )
      : null;
    const ranks = rankRes?.ok ? await rankRes.json() : [];

    // ウォッチリスト取得
    const watchlist = await getWatchlist(userId);

    const parts: string[] = [];

    // N225シグナル
    const n225Lines: string[] = [];
    if (n225.length > 0) {
      n225Lines.push(`日経225: ${Number(n225[0].close).toLocaleString()}円（${n225[0].date}）`);
    }
    if (ranks.length > 0) {
      const dps = ranks.map((r: { drop_prob: number }) => r.drop_prob).filter((d: number) => d != null);
      if (dps.length > 0) {
        const avg = dps.reduce((a: number, b: number) => a + b, 0) / dps.length;
        const signal = avg >= 15 ? "🔴 キャッシュ推奨" : "🟢 投資継続OK";
        n225Lines.push(`市場平均下落確率: ${avg.toFixed(1)}% → ${signal}`);
      }
    }
    if (n225Lines.length > 0) parts.push(`📊 N225シグナル（${latestDate ?? "最新"}）\n` + n225Lines.join("\n"));

    // ウォッチリスト状況
    if (watchlist.length > 0) {
      const wLines: string[] = ["📋 ウォッチリスト状況:"];
      for (const w of watchlist) {
        const stock = await lookupStock(w.code);
        if (!stock) { wLines.push(`  ${w.name}(${w.code}): データなし`); continue; }
        const dp = stock.drop_prob;
        const buyTh = w.dp_threshold ?? 8.0;
        const sellTh = w.dp_sell_threshold ?? 20.0;
        const mark = dp < buyTh ? " 🔔買い時！" : dp >= sellTh ? " ⚠️売り検討" : "";
        wLines.push(`  ${stock.name}(${w.code}) ${Number(stock.close).toLocaleString()}円 下落確率${dp}%${mark}`);
      }
      parts.push(wLines.join("\n"));
    }

    return { handled: true, reply: parts.length > 0 ? parts.join("\n\n") : "データを取得できませんでした。" };
  }

  if (/^(ウォッチ|リスト|一覧)$/i.test(trimmed)) {
    const list = await getWatchlist(userId);
    if (list.length === 0) {
      return { handled: true, reply: "ウォッチリストは空です。\n「ウォッチ 8473」で追加できます。" };
    }
    const lines = ["📋 ウォッチリスト:"];
    for (const item of list) {
      const stock = await lookupStock(item.code);
      const dpStr = stock ? `下落確率${stock.drop_prob}%` : "";
      const priceStr = stock ? `${stock.close.toLocaleString()}円` : "";
      let status = "";
      if (stock) {
        if (stock.drop_prob < item.dp_threshold) status = "🔔買い時！";
        else if (stock.drop_prob >= (item.dp_sell_threshold ?? 20)) status = "⚠️売り検討";
      }
      lines.push(`  ${item.code} ${item.name} ${priceStr} ${dpStr} ${status}`);
    }
    lines.push(`\n※ 下落確率が買い閾値未満で買い通知、売り閾値以上で売り通知`);
    return { handled: true, reply: lines.join("\n") };
  }

  const addMatch = trimmed.match(
    /^ウォッチ\s+(.+?)(?:\s+(\d+(?:\.\d+)?))?(?:\s+(\d+(?:\.\d+)?))?$/i,
  );
  if (addMatch) {
    const premium = user ? isPremium(user) : false;
    if (!premium) {
      const currentList = await getWatchlist(userId);
      if (currentList.length >= FREE_WATCHLIST_LIMIT) {
        return { handled: true, reply: `⚠️ 無料プランのウォッチリスト上限（${FREE_WATCHLIST_LIMIT}銘柄）に達しています。\n「解除 コード」で空きを作るか、プレミアムプランで無制限にできます。` };
      }
    }
    const target = addMatch[1].trim();
    const threshold = addMatch[2] ? parseFloat(addMatch[2]) : 8.0;
    const sellThreshold = addMatch[3] ? parseFloat(addMatch[3]) : 20.0;

    if (/^\d{4}$/.test(target)) {
      const stock = await lookupStock(target);
      if (!stock) {
        return { handled: true, reply: `${target} はランキングに見つかりません。銘柄コード4桁で指定してください。` };
      }
      await addToWatchlist(userId, stock.code, stock.name, threshold, sellThreshold);
      return {
        handled: true,
        reply:
          `✅ ${stock.name}(${stock.code}) をウォッチリストに追加\n` +
          `株価: ${stock.close.toLocaleString()}円\n` +
          `下落確率: ${stock.drop_prob}%\n` +
          `\n※ 下落確率 < ${threshold}%で買い通知 / ≥ ${sellThreshold}%で売り通知`,
      };
    }

    const matches = await searchStockByName(target);
    if (matches.length === 0) {
      return { handled: true, reply: `「${target}」に該当する銘柄が見つかりません。コード4桁で指定してください。` };
    }
    if (matches.length === 1) {
      const stock = await lookupStock(matches[0].code);
      const name = stock?.name ?? matches[0].name;
      await addToWatchlist(userId, matches[0].code, name, threshold, sellThreshold);
      return {
        handled: true,
        reply:
          `✅ ${name}(${matches[0].code}) をウォッチリストに追加\n` +
          (stock ? `株価: ${stock.close.toLocaleString()}円\n下落確率: ${stock.drop_prob}%\n` : "") +
          `\n※ 下落確率 < ${threshold}%で買い通知 / ≥ ${sellThreshold}%で売り通知`,
      };
    }
    const options = matches.map((m) => `  ${m.code} ${m.name}`).join("\n");
    return {
      handled: true,
      reply: `「${target}」に複数該当:\n${options}\n\nコードで指定してください。\n例: ウォッチ ${matches[0].code}`,
    };
  }

  const removeMatch = trimmed.match(/^(解除|削除|外す)\s+(\d{4})$/i);
  if (removeMatch) {
    const code = removeMatch[2];
    await removeFromWatchlist(userId, code);
    return { handled: true, reply: `🗑 ${code} をウォッチリストから削除しました。` };
  }

  // 「7203 推移」「7203の下落確率履歴」など — 時系列専用
  const historyMatch = trimmed.match(/^(\d{4})\s*(推移|履歴|時系列|history)/);
  if (historyMatch) {
    const hist = await fetchDropProbHistory(historyMatch[1], 30);
    return { handled: true, reply: hist };
  }

  if (/^\d{4}$/.test(trimmed)) {
    const stock = await lookupStock(trimmed);
    if (!stock) {
      return { handled: true, reply: `${trimmed} のデータが見つかりません。` };
    }
    const dpStatus =
      stock.drop_prob < 8 ? "🟢安全圏" : stock.drop_prob >= 15 ? "🔴危険" : "🟡通常";
    return {
      handled: true,
      reply:
        `📊 ${stock.name}(${stock.code})\n` +
        `株価: ${stock.close.toLocaleString()}円\n` +
        `下落確率: ${stock.drop_prob}% ${dpStatus}\n\n` +
        `「${stock.code} 推移」で過去30日の下落確率推移を表示\n` +
        `「ウォッチ ${stock.code}」でウォッチに追加`,
    };
  }

  return { handled: false };
}

// ── 会話履歴 ─────────────────────────────────

async function saveMessage(userId: string, role: "user" | "assistant", content: string): Promise<void> {
  if (!SB_URL || !SB_KEY) return;
  await fetch(`${SB_URL}/rest/v1/line_chat_history`, {
    method: "POST",
    headers: sbHeaders({ Prefer: "return=minimal" }),
    body: JSON.stringify({ line_user_id: userId, role, content: content.slice(0, 2000) }),
  });
}

async function trimHistory(userId: string): Promise<void> {
  if (!SB_URL || !SB_KEY) return;
  const res = await fetch(
    `${SB_URL}/rest/v1/line_chat_history?line_user_id=eq.${userId}&order=created_at.desc&select=id&offset=6&limit=100`,
    { headers: sbHeaders() },
  );
  if (res.ok) {
    const old = await res.json();
    if (old.length > 0) {
      const ids = old.map((r: { id: string }) => r.id);
      await fetch(`${SB_URL}/rest/v1/line_chat_history?id=in.(${ids.join(",")})`, {
        method: "DELETE",
        headers: sbHeaders(),
      });
    }
  }
}

async function getRecentMessages(userId: string): Promise<{ role: string; content: string }[]> {
  if (!SB_URL || !SB_KEY) return [];
  const res = await fetch(
    `${SB_URL}/rest/v1/line_chat_history?line_user_id=eq.${userId}&order=created_at.desc&limit=6&select=role,content`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return [];
  const rows = await res.json();
  return rows.reverse();
}

// ── 企業情報データ取得 ────────────────────────

async function fetchTdnetDisclosures(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";
  const res = await fetch(
    `${SB_URL}/rest/v1/ext_tdnet_disclosures?code=eq.${code}&order=disclosed_at.desc&select=disclosed_at,title,category&limit=5`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "";
  const rows = await res.json();
  if (rows.length === 0) return "";
  const lines = ["【適時開示（TDnet）】"];
  for (const r of rows) {
    const d = r.disclosed_at?.slice(0, 10) ?? "";
    const cat = r.category ? `[${r.category}]` : "";
    lines.push(`  ${d} ${cat} ${r.title}`);
  }
  return lines.join("\n");
}

async function fetchJpxShortSelling(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";
  const res = await fetch(
    `${SB_URL}/rest/v1/jpx_short_selling?code=eq.${code}&order=calc_date.desc&select=calc_date,short_seller,short_ratio&limit=5`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "";
  const rows = await res.json();
  if (rows.length === 0) return "";
  const lines = ["【空売り残高（JPX）】"];
  for (const r of rows) {
    lines.push(`  ${r.calc_date} ${r.short_seller}: ${r.short_ratio}%`);
  }
  return lines.join("\n");
}

async function fetchJpxMarginBalance(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";
  const res = await fetch(
    `${SB_URL}/rest/v1/jpx_margin_balance?code=eq.${code}&order=record_date.desc&select=record_date,margin_buy,margin_sell&limit=3`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "";
  const rows = await res.json();
  if (rows.length === 0) return "";
  const lines = ["【信用残高（JPX）】"];
  for (const r of rows) {
    lines.push(`  ${r.record_date} 買残: ${Number(r.margin_buy).toLocaleString()} / 売残: ${Number(r.margin_sell).toLocaleString()}`);
  }
  return lines.join("\n");
}

// ── 需給シグナル要約 ────────────────────────

function summarizeSupplyDemand(
  shortRows: { calc_date: string; short_seller: string; short_ratio: number }[],
  marginRows: { record_date: string; margin_buy: number; margin_sell: number }[],
): string {
  const signals: string[] = [];

  if (shortRows.length >= 2) {
    const totalLatest = shortRows.filter((r) => r.calc_date === shortRows[0].calc_date)
      .reduce((s, r) => s + Number(r.short_ratio), 0);
    const nextDate = shortRows.find((r) => r.calc_date !== shortRows[0].calc_date)?.calc_date;
    if (nextDate) {
      const totalPrev = shortRows.filter((r) => r.calc_date === nextDate)
        .reduce((s, r) => s + Number(r.short_ratio), 0);
      if (totalLatest > totalPrev * 1.2) {
        signals.push("🔺 空売り急増 → 下落圧力・ただし踏み上げ警戒");
      } else if (totalLatest < totalPrev * 0.8) {
        signals.push("🔻 空売り減少 → ショートカバー（買戻し圧力）");
      }
    }
    if (totalLatest > 5) {
      signals.push(`⚠️ 空売り比率 ${totalLatest.toFixed(1)}% → 高水準（踏み上げ or 続落に注意）`);
    }
  }

  if (marginRows.length >= 2) {
    const latest = marginRows[0];
    const prev = marginRows[1];
    const buyChange = (Number(latest.margin_buy) - Number(prev.margin_buy)) / Math.max(Number(prev.margin_buy), 1);
    const sellChange = (Number(latest.margin_sell) - Number(prev.margin_sell)) / Math.max(Number(prev.margin_sell), 1);
    const ratio = Number(latest.margin_sell) > 0 ? Number(latest.margin_buy) / Number(latest.margin_sell) : null;

    if (buyChange > 0.1) signals.push("📈 信用買い増加 → 個人の強気姿勢");
    if (buyChange < -0.1) signals.push("📉 信用買い減少 → 個人の手仕舞い");
    if (sellChange > 0.1) signals.push("📈 信用売り増加 → 弱気ヘッジ増");
    if (sellChange < -0.1) signals.push("📉 信用売り減少 → 買戻し圧力");
    if (ratio != null) {
      if (ratio > 5) signals.push(`⚠️ 貸借倍率 ${ratio.toFixed(1)}倍 → 買い偏り（将来の売り圧力）`);
      else if (ratio < 1) signals.push(`💡 貸借倍率 ${ratio.toFixed(1)}倍 → 売り優勢（逆張り妙味）`);
    }
  }

  if (signals.length === 0) return "";
  const latestDate = shortRows.length > 0 ? shortRows[0].calc_date : "";
  const dateStr = latestDate ? `（${latestDate}時点）` : "";
  return `【需給シグナル${dateStr}】\n` + signals.map((s) => `  ${s}`).join("\n");
}

// ── セクター横断比較 ────────────────────────

async function fetchSectorComparison(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const sectorRes = await fetch(
    `${SB_URL}/rest/v1/jpx_stock_list?code=eq.${code}&select=sector`,
    { headers: sbHeaders() },
  );
  if (!sectorRes.ok) return "";
  const sectorRows = await sectorRes.json();
  if (sectorRows.length === 0 || !sectorRows[0].sector) return "";
  const sector = sectorRows[0].sector;

  const today = new Date().toISOString().slice(0, 10);
  const peerRes = await fetch(
    `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob,per,pbr&order=drop_prob.asc`,
    { headers: sbHeaders() },
  );
  if (!peerRes.ok) return "";
  const allRanks = await peerRes.json();

  const codeListRes = await fetch(
    `${SB_URL}/rest/v1/jpx_stock_list?sector=eq.${encodeURIComponent(sector)}&select=code`,
    { headers: sbHeaders() },
  );
  if (!codeListRes.ok) return "";
  const sectorCodes = new Set((await codeListRes.json()).map((r: { code: string }) => r.code));

  const peers = allRanks.filter((r: { code: string }) => sectorCodes.has(r.code));
  if (peers.length < 2) return "";

  const target = peers.find((r: { code: string }) => r.code === code);
  const rank = target ? peers.indexOf(target) + 1 : null;
  const avgDp = peers.reduce((s: number, r: { drop_prob: number }) => s + r.drop_prob, 0) / peers.length;
  const avgPer = peers.filter((r: { per: number | null }) => r.per != null && r.per > 0);
  const avgPerVal = avgPer.length > 0 ? avgPer.reduce((s: number, r: { per: number }) => s + r.per, 0) / avgPer.length : null;

  const lines = [`【セクター比較: ${sector}（${peers.length}社）】`];
  if (rank != null) {
    lines.push(`  安全性順位: ${rank}位/${peers.length}社`);
  }
  lines.push(`  セクター平均下落確率: ${avgDp.toFixed(1)}%`);
  if (avgPerVal != null && target?.per) {
    const vs = target.per < avgPerVal ? "割安" : "割高";
    lines.push(`  セクター平均PER: ${avgPerVal.toFixed(1)} → 当銘柄${target.per}は${vs}`);
  }

  const top3 = peers.slice(0, 3);
  if (top3.length > 0) {
    lines.push(`  同セクター安全Top3:`);
    for (const p of top3) {
      lines.push(`    ${p.code} ${p.name}: 下落確率${p.drop_prob}% PER=${p.per ?? "N/A"}`);
    }
  }

  return lines.join("\n");
}

// ── 市場データ取得 ───────────────────────────

async function fetchStockDetail(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const [rankRes, finRes, tdnetStr, shortRes, marginRes, sectorStr] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?code=eq.${code}&select=date,code,name,close,drop_prob,per,pbr,piotroski,bps_growth,eps_surprise,vol,rel20&order=date.desc&limit=30`,
      { headers: sbHeaders() },
    ),
    fetch(
      `${SB_URL}/rest/v1/jquants_fin_summary?code=eq.${code}&select=code,disc_date,doc_type,eps,bps,div_ann,payout_ratio,np,cfo,ta,equity,op,sales,fnp,fop,fsales&order=disc_date.desc&limit=8`,
      { headers: sbHeaders() },
    ),
    fetchTdnetDisclosures(code),
    fetch(
      `${SB_URL}/rest/v1/jpx_short_selling?code=eq.${code}&order=calc_date.desc&select=calc_date,short_seller,short_ratio&limit=10`,
      { headers: sbHeaders() },
    ),
    fetch(
      `${SB_URL}/rest/v1/jpx_margin_balance?code=eq.${code}&order=record_date.desc&select=record_date,margin_buy,margin_sell&limit=3`,
      { headers: sbHeaders() },
    ),
    fetchSectorComparison(code),
  ]);

  const ranks = rankRes.ok ? await rankRes.json() : [];
  const fins = finRes.ok ? await finRes.json() : [];
  const shortRows = shortRes.ok ? await shortRes.json() : [];
  const marginRows = marginRes.ok ? await marginRes.json() : [];
  const lines: string[] = [];

  if (ranks.length > 0) {
    const r = ranks[0];
    lines.push(`【${r.name}(${r.code})の詳細データ】`);
    // ranksはdate.desc（[0]=最新）
    const changes = computePriceChanges(ranks);
    const changeStr = changes.length > 0 ? `（${changes.join(", ")}）` : "";
    lines.push(`株価: ${r.close}円${changeStr}, 下落確率: ${r.drop_prob}%`);
    lines.push(`PER（株価収益率＝株価÷EPS。低いほど割安）: ${r.per}（目安<15）, PBR（株価純資産倍率＝株価÷BPS。純資産の何倍で買われているか）: ${r.pbr}（目安<1）, Piotroski: ${r.piotroski}`);
    lines.push(`BPS成長: ${r.bps_growth}, EPSサプライズ: ${r.eps_surprise}`);
    lines.push(`出来高: ${r.vol}, 20日相対強度: ${r.rel20}`);
    if (ranks.length > 1) {
      // 古い順に並べ直して時系列表示（最大10件）
      const history = [...ranks].reverse().slice(-10);
      const trend = history.map((x: { date: string; drop_prob: number }) => {
        const d = x.date ? x.date.slice(5) : ""; // MM-DD
        return `${d}:${x.drop_prob}%`;
      }).join(" → ");
      lines.push(`\n【下落確率の推移(直近${history.length}日)】\n${trend}`);
    }
  }
  if (fins.length > 0) {
    const f = fins[0];
    // 配当は通期(FY)決算にのみ記録される。中間期はnullなので最新FYを優先して使う
    const fyFin = fins.find((x: { doc_type: string; div_ann: number | null }) => x.doc_type === "FY" && x.div_ann != null) ?? fins.find((x: { div_ann: number | null }) => x.div_ann != null) ?? f;
    const close = ranks.length > 0 ? Number(ranks[0].close) : 0;
    const divYield = close > 0 && fyFin.div_ann ? ((fyFin.div_ann / close) * 100).toFixed(2) : null;
    const roe = f.equity && f.equity > 0 && f.np ? ((f.np / f.equity) * 100).toFixed(1) : null;
    const equityRatio = f.ta && f.ta > 0 && f.equity ? ((f.equity / f.ta) * 100).toFixed(1) : null;
    const opMargin = f.sales && f.sales > 0 && f.op ? ((f.op / f.sales) * 100).toFixed(1) : null;

    lines.push(`\n【決算データ(${f.disc_date} ${f.doc_type})】`);
    lines.push(`EPS: ${f.eps}, BPS: ${f.bps}`);
    if (divYield)
      lines.push(
        `配当利回り（年間配当÷株価。高いほど配当収入が多い）: ${divYield}%（目安>3%）, 年間配当: ${fyFin.div_ann}円, 配当性向（配当÷純利益。高すぎると継続困難）: ${fyFin.payout_ratio ? (fyFin.payout_ratio * 100).toFixed(1) + "%" : "N/A"}（目安<70%）`,
      );
    if (roe) lines.push(`ROE（自己資本利益率＝純利益÷自己資本。株主資本の稼ぎ効率）: ${roe}%（目安>10%）`);
    if (equityRatio) lines.push(`自己資本比率: ${equityRatio}%`);
    if (opMargin) lines.push(`営業利益率: ${opMargin}%`);
    if (f.cfo != null) lines.push(`営業CF: ${(f.cfo / 1e6).toFixed(0)}百万円`);
    if (f.fnp != null && f.np != null && f.fnp > 0) {
      const progress = ((f.np / f.fnp) * 100).toFixed(1);
      lines.push(
        `通期予想進捗率: ${progress}% (実績NP: ${(f.np / 1e6).toFixed(0)}百万 / 予想: ${(f.fnp / 1e6).toFixed(0)}百万)`,
      );
    }

    const fyRecords = fins.filter((x: { doc_type: string }) => x.doc_type === "FY");
    if (fyRecords.length >= 2) {
      const cur = fyRecords[0];
      const prev = fyRecords[1];
      if (cur.np != null && prev.np != null && prev.np !== 0) {
        const npGrowth = (((cur.np - prev.np) / Math.abs(prev.np)) * 100).toFixed(1);
        lines.push(`純利益成長率(前期比): ${npGrowth}%`);
      }
      if (cur.sales != null && prev.sales != null && prev.sales !== 0) {
        const salesGrowth = (((cur.sales - prev.sales) / Math.abs(prev.sales)) * 100).toFixed(1);
        lines.push(`売上成長率(前期比): ${salesGrowth}%`);
      }
    }
  }

  if (tdnetStr) lines.push("\n" + tdnetStr);

  // 需給生データ表示
  if (shortRows.length > 0) {
    const shortLines = ["【空売り残高（JPX）】"];
    for (const r of shortRows.slice(0, 5)) {
      shortLines.push(`  ${r.calc_date} ${r.short_seller}: ${r.short_ratio}%`);
    }
    lines.push("\n" + shortLines.join("\n"));
  }
  if (marginRows.length > 0) {
    const marginLines = ["【信用残高（JPX）】"];
    for (const r of marginRows) {
      marginLines.push(`  ${r.record_date} 買残: ${Number(r.margin_buy).toLocaleString()} / 売残: ${Number(r.margin_sell).toLocaleString()}`);
    }
    lines.push("\n" + marginLines.join("\n"));
  }

  // 需給シグナル要約
  const supplyDemandSignal = summarizeSupplyDemand(shortRows, marginRows);
  if (supplyDemandSignal) lines.push("\n" + supplyDemandSignal);

  // セクター横断比較
  if (sectorStr) lines.push("\n" + sectorStr);

  return lines.join("\n");
}

async function fetchMarketContext(userId: string, userMessage: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const today = new Date().toISOString().slice(0, 10);

  const [rankRes, n225Res, watchRes] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob,per,pbr&order=drop_prob.asc&limit=20`,
      { headers: sbHeaders() },
    ),
    fetch(
      `${SB_URL}/rest/v1/yahoo_market_index?ticker=eq.N225&order=date.desc&limit=1&select=date,close`,
      { headers: sbHeaders() },
    ),
    fetch(
      `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold,dp_sell_threshold`,
      { headers: sbHeaders() },
    ),
  ]);

  const rankings = rankRes.ok ? await rankRes.json() : [];
  const n225 = n225Res.ok ? await n225Res.json() : [];
  const watchlist: WatchItem[] = watchRes.ok ? await watchRes.json() : [];

  const lines: string[] = [];

  if (n225.length > 0) {
    lines.push(`N225: ${n225[0].close}円 (${n225[0].date})`);
  }

  if (rankings.length > 0) {
    const dps = rankings
      .filter((r: { drop_prob: number | null }) => r.drop_prob != null)
      .map((r: { drop_prob: number }) => r.drop_prob);
    const avgDp = dps.length > 0 ? dps.reduce((a: number, b: number) => a + b, 0) / dps.length : null;

    if (avgDp != null) {
      const signal = avgDp >= 15 ? "🔴キャッシュ推奨" : "🟢投資継続OK";
      lines.push(`市場平均下落確率: ${avgDp.toFixed(1)}% → ${signal}`);
    }

    lines.push(`\n本日のトップ10:`);
    for (const r of rankings.slice(0, 10)) {
      lines.push(`  ${r.code} ${r.name}: 下落確率=${r.drop_prob}% PER=${r.per} PBR=${r.pbr}`);
    }
  }

  if (watchlist.length > 0) {
    // ウォッチリスト銘柄の直近6営業日分を取得（前日比=直近2日、前週比=直近6営業日≒1週間で算出）
    const watchCodes = watchlist.map((w) => w.code);
    const trendRes = await fetch(
      `${SB_URL}/rest/v1/gen_rankings?code=in.(${watchCodes.join(",")})&select=code,date,close,drop_prob&order=code.asc,date.desc`,
      { headers: sbHeaders() },
    );
    const trendRows: { code: string; date: string; close: number; drop_prob: number }[] = trendRes.ok ? await trendRes.json() : [];

    const TREND_DAYS = 6;
    const trendByCode = new Map<string, { date: string; drop_prob: number; close: number }[]>();
    for (const r of trendRows) {
      if (!trendByCode.has(r.code)) trendByCode.set(r.code, []);
      const arr = trendByCode.get(r.code)!;
      if (arr.length < TREND_DAYS) arr.push({ date: r.date, drop_prob: r.drop_prob, close: r.close });
    }

    lines.push(`\nこのユーザーのウォッチリスト:`);
    for (const w of watchlist) {
      const trend = trendByCode.get(w.code) ?? []; // 新しい順（[0]=最新）
      const latest = trend[0];
      const dpStr = latest ? `下落確率=${latest.drop_prob}%` : "下落確率=--";
      let closeStr = "";
      if (latest) {
        const changes = computePriceChanges(trend).join(", ");
        closeStr = ` 株価=${latest.close}円${changes ? `(${changes})` : ""}`;
      }
      // 直近日数分の推移（古い→新しい順）
      const trendStr = trend.length >= 2
        ? ` 推移(古→新)=${[...trend].reverse().map((t) => `${t.drop_prob}%`).join("→")}`
        : "";
      lines.push(`  ${w.code} ${w.name}:${closeStr} ${dpStr}${trendStr} (買<${w.dp_threshold} 売≥${w.dp_sell_threshold ?? 20})`);
    }
  }

  const codeMatch = userMessage.match(/(\d{4})/);
  if (codeMatch) {
    const detail = await fetchStockDetail(codeMatch[1]);
    if (detail) lines.push("\n" + detail);
  }

  return lines.join("\n");
}

// ── Claude API (tool use) ─────────────────────

const TOOLS: Anthropic.Tool[] = [
  {
    name: "add_watchlist",
    description:
      "ウォッチリストに銘柄を追加する。ユーザーが「〇〇をウォッチして」「〇〇を監視したい」「〇〇が下がったら教えて」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "銘柄コード(4桁)または銘柄名の一部。例: '8473', 'SBI', '三菱UFJ'" },
        buy_threshold: { type: "number", description: "買い閾値(dp%)。指定なければ8.0" },
        sell_threshold: { type: "number", description: "売り閾値(dp%)。指定なければ20.0" },
      },
      required: ["query"],
    },
  },
  {
    name: "remove_watchlist",
    description: "ウォッチリストから銘柄を削除する。「〇〇を外して」「〇〇の監視やめて」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "銘柄コード(4桁)または銘柄名の一部" },
      },
      required: ["query"],
    },
  },
  {
    name: "show_watchlist",
    description: "ウォッチリストの一覧を表示する。「ウォッチリスト見せて」「今何を監視してる？」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {},
    },
  },
  {
    name: "lookup_stock",
    description: "特定銘柄の現在の詳細情報を調べる。「〇〇の状況は？」「〇〇って今どう？」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "銘柄コード(4桁)または銘柄名の一部" },
      },
      required: ["query"],
    },
  },
  {
    name: "screen_stocks",
    description: "条件に合う銘柄をスクリーニングする。「高配当で安全な銘柄」「PBR1倍割れ」「下落確率が低い割安株」「配当利回り3%以上」等と言ったとき使う。自然言語の条件をフィルターパラメータに変換して呼ぶ。",
    input_schema: {
      type: "object" as const,
      properties: {
        max_drop_prob: { type: "number", description: "下落確率の上限(%)。安全な銘柄=8、普通以下=15" },
        min_div_yield: { type: "number", description: "最低配当利回り(%)。高配当=3.0" },
        max_per: { type: "number", description: "PER上限。割安=15" },
        max_pbr: { type: "number", description: "PBR上限。純資産割れ=1.0" },
        min_piotroski: { type: "number", description: "Piotroski最低点(0-9)。財務健全=7" },
        min_roe: { type: "number", description: "ROE下限(%)。良好=10" },
        sector: { type: "string", description: "セクター名（日本語）で絞り込み。例: '情報・通信業', '銀行業'" },
        sort_by: { type: "string", description: "ソート基準。'drop_prob'(安全順), 'div_yield'(高配当順), 'per'(割安順), 'pbr'(PBR順)。div_yieldは決算データ結合のため若干遅い", enum: ["drop_prob", "div_yield", "per", "pbr"] },
        limit: { type: "number", description: "表示件数。デフォルト5" },
      },
    },
  },
  {
    name: "check_catalyst",
    description: "直近の決算サプライズ・適時開示カタリスト・EDINET大量保有報告を確認する。「最近の決算サプライズは？」「ウォッチ銘柄のニュースは？」「上方修正した銘柄は？」「増配銘柄」「大量保有」「機関投資家の動き」「5%保有報告」等と言ったとき使う。ウォッチリスト銘柄のカタリストも自動チェックする。「ウォッチ銘柄の」「保有株の」等と明示されない限り、scope='all'（市場全体）を既定にする。「ほかにも」「他の銘柄は」等ウォッチリストに限定しない追加質問も必ずscope='all'で呼び直す。",
    input_schema: {
      type: "object" as const,
      properties: {
        scope: { type: "string", description: "'watchlist'=ウォッチリスト銘柄のみに限定する場合だけ指定。それ以外は'all'（全銘柄から検索、既定）", enum: ["watchlist", "all"] },
        keyword: { type: "string", description: "検索キーワード。例: '上方修正', '増配', '自社株買い', '株式分割'" },
        days: { type: "number", description: "何日前までの開示を検索するか。デフォルト7" },
      },
    },
  },
];

async function resolveCode(
  query: string,
): Promise<{ code: string; name: string } | { candidates: { code: string; name: string }[] } | null> {
  if (/^\d{4}$/.test(query)) {
    const stock = await lookupStock(query);
    return stock ? { code: stock.code, name: stock.name } : null;
  }
  const matches = await searchStockByName(query);
  if (matches.length === 0) return null;
  if (matches.length === 1) return matches[0];
  return { candidates: matches };
}

async function executeToolCall(
  toolName: string,
  input: Record<string, unknown>,
  userId: string,
): Promise<string> {
  if (toolName === "show_watchlist") {
    const list = await getWatchlist(userId);
    if (list.length === 0) return "ウォッチリストは空です。銘柄名やコードを言ってもらえれば追加します。";
    const lines = ["📋 ウォッチリスト:"];
    for (const item of list) {
      const stock = await lookupStock(item.code);
      const dpStr = stock ? `下落確率${stock.drop_prob}%` : "";
      const priceStr = stock ? `${stock.close.toLocaleString()}円` : "";
      let status = "";
      if (stock) {
        if (stock.drop_prob < item.dp_threshold) status = "🔔買い時！";
        else if (stock.drop_prob >= (item.dp_sell_threshold ?? 20)) status = "⚠️売り検討";
      }
      lines.push(`  ${item.code} ${item.name} ${priceStr} ${dpStr} ${status}`);
    }
    lines.push(`\n※ 下落確率が買い閾値未満で買い通知、売り閾値以上で売り通知`);
    return lines.join("\n");
  }

  if (toolName === "add_watchlist") {
    const resolved = await resolveCode(input.query as string);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    const buyTh = (input.buy_threshold as number) ?? 8.0;
    const sellTh = (input.sell_threshold as number) ?? 20.0;
    await addToWatchlist(userId, resolved.code, resolved.name, buyTh, sellTh);
    const stock = await lookupStock(resolved.code);
    return (
      `✅ ${resolved.name}(${resolved.code}) をウォッチリストに追加` +
      (stock ? `\n株価: ${stock.close.toLocaleString()}円\n下落確率: ${stock.drop_prob}%` : "") +
      `\n\n※ 下落確率 < ${buyTh}%で買い通知 / ≥ ${sellTh}%で売り通知`
    );
  }

  if (toolName === "remove_watchlist") {
    const resolved = await resolveCode(input.query as string);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    await removeFromWatchlist(userId, resolved.code);
    return `🗑 ${resolved.name}(${resolved.code}) をウォッチリストから削除しました。`;
  }

  if (toolName === "lookup_stock") {
    const resolved = await resolveCode(input.query as string);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    const detail = await fetchStockDetail(resolved.code);
    return detail || `${resolved.code} のデータが見つかりません。`;
  }

  if (toolName === "screen_stocks") {
    return await executeScreenStocks(input, userId);
  }

  if (toolName === "check_catalyst") {
    return await executeCheckCatalyst(input, userId);
  }

  return "不明な操作です。";
}

// ── スクリーナー実行 ────────────────────────
async function executeScreenStocks(input: Record<string, unknown>, _userId: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "データベースに接続できません。";
  const today = new Date().toISOString().slice(0, 10);

  const sortBy = (input.sort_by as string) || "drop_prob";
  const limit = Math.min((input.limit as number) || 5, 10);
  const orderCol = `${sortBy === "div_yield" ? "drop_prob" : sortBy}.asc`;

  const rankRes = await fetch(
    `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob,per,pbr,piotroski,bps_growth,eps_surprise&order=${orderCol}&limit=500`,
    { headers: sbHeaders() },
  );
  if (!rankRes.ok) return "ランキングデータの取得に失敗しました。";
  let rows: Record<string, unknown>[] = await rankRes.json();

  const maxDp = input.max_drop_prob as number | undefined;
  const minDy = input.min_div_yield as number | undefined;
  const maxPer = input.max_per as number | undefined;
  const maxPbr = input.max_pbr as number | undefined;
  const minPio = input.min_piotroski as number | undefined;
  const minRoe = input.min_roe as number | undefined;
  const sector = input.sector as string | undefined;

  if (maxDp != null) rows = rows.filter((r) => r.drop_prob != null && (r.drop_prob as number) <= maxDp);
  if (maxPer != null) rows = rows.filter((r) => r.per != null && (r.per as number) > 0 && (r.per as number) <= maxPer);
  if (maxPbr != null) rows = rows.filter((r) => r.pbr != null && (r.pbr as number) > 0 && (r.pbr as number) <= maxPbr);
  if (minPio != null) rows = rows.filter((r) => r.piotroski != null && (r.piotroski as number) >= minPio);

  if (sector) {
    const codeListRes = await fetch(
      `${SB_URL}/rest/v1/jpx_stock_list?sector=eq.${encodeURIComponent(sector)}&select=code`,
      { headers: sbHeaders() },
    );
    if (codeListRes.ok) {
      const sectorCodes = new Set((await codeListRes.json()).map((r: { code: string }) => r.code));
      rows = rows.filter((r) => sectorCodes.has(r.code as string));
    }
  }

  // 配当利回り・ROEが必要な場合はjquants_fin_summaryから取得
  if (minDy != null || minRoe != null || sortBy === "div_yield") {
    const codes = rows.map((r) => r.code as string);
    if (codes.length > 0) {
      const finRes = await fetch(
        `${SB_URL}/rest/v1/jquants_fin_summary?code=in.(${codes.join(",")})&select=code,doc_type,div_ann,np,equity&order=disc_date.desc`,
        { headers: sbHeaders() },
      );
      if (finRes.ok) {
        const fins: Record<string, unknown>[] = await finRes.json();
        // 銘柄ごとに最新FY（配当あり）を優先、なければ最新レコードを使う
        const byCode = new Map<string, Record<string, unknown>[]>();
        for (const f of fins) {
          const c = f.code as string;
          if (!byCode.has(c)) byCode.set(c, []);
          byCode.get(c)!.push(f);
        }
        const finMap = new Map<string, { div_yield: number | null; roe: number | null }>();
        for (const [c, recs] of byCode.entries()) {
          const best = recs.find((x) => x.doc_type === "FY" && x.div_ann != null)
            ?? recs.find((x) => x.div_ann != null)
            ?? recs[0];
          const close = (rows.find((r) => r.code === c)?.close as number) || 0;
          const dy = close > 0 && best.div_ann ? ((best.div_ann as number) / close) * 100 : null;
          const roe = best.equity && (best.equity as number) > 0 && best.np ? ((best.np as number) / (best.equity as number)) * 100 : null;
          finMap.set(c, { div_yield: dy, roe });
        }
        for (const r of rows) {
          const fm = finMap.get(r.code as string);
          if (fm) { r._div_yield = fm.div_yield; r._roe = fm.roe; }
        }
        if (minDy != null) rows = rows.filter((r) => r._div_yield != null && (r._div_yield as number) >= minDy);
        if (minRoe != null) rows = rows.filter((r) => r._roe != null && (r._roe as number) >= minRoe);
        if (sortBy === "div_yield") rows.sort((a, b) => ((b._div_yield as number) || 0) - ((a._div_yield as number) || 0));
      }
    }
  }

  if (rows.length === 0) return "条件に合う銘柄が見つかりません。条件を緩めてみてください。";

  const result = rows.slice(0, limit);
  const filters: string[] = [];
  if (maxDp != null) filters.push(`下落確率≤${maxDp}%`);
  if (minDy != null) filters.push(`配当利回り≥${minDy}%`);
  if (maxPer != null) filters.push(`PER≤${maxPer}`);
  if (maxPbr != null) filters.push(`PBR≤${maxPbr}`);
  if (minPio != null) filters.push(`Piotroski≥${minPio}`);
  if (minRoe != null) filters.push(`ROE≥${minRoe}%`);
  if (sector) filters.push(`セクター:${sector}`);

  const lines = [`🔍 スクリーニング結果（${filters.join(" / ")}）\n該当: ${rows.length}銘柄中 上位${result.length}件\n`];
  for (let i = 0; i < result.length; i++) {
    const r = result[i];
    const dp = r.drop_prob as number;
    const dpStatus = dp < 8 ? "🟢" : dp >= 15 ? "🔴" : "🟡";
    const dyStr = r._div_yield != null ? (r._div_yield as number).toFixed(1) + "%" : "N/A";
    lines.push(
      `${i + 1}. ${r.code} ${r.name}` +
      `\n   ${(r.close as number)?.toLocaleString() ?? "?"}円 下落確率${dp}%${dpStatus}` +
      `\n   PER=${r.per ?? "N/A"} PBR=${r.pbr ?? "N/A"} 配当=${dyStr} Pio=${r.piotroski ?? "N/A"}`,
    );
  }
  return lines.join("\n");
}

// ── カタリスト検出 ────────────────────────
async function executeCheckCatalyst(input: Record<string, unknown>, userId: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "データベースに接続できません。";

  const scope = (input.scope as string) || "all";
  const keyword = input.keyword as string | undefined;
  const days = Math.min((input.days as number) || 7, 30);

  const since = new Date();
  since.setDate(since.getDate() - days);
  const sinceStr = since.toISOString().slice(0, 10);

  let codeFilter = "";
  let watchCodes: string[] = [];
  if (scope === "watchlist") {
    const list = await getWatchlist(userId);
    if (list.length === 0) return "ウォッチリストが空です。先に銘柄を追加してください。";
    watchCodes = list.map((w) => w.code);
    codeFilter = `&code=in.(${watchCodes.join(",")})`;
  }

  let titleFilter = "";
  if (keyword) {
    titleFilter = `&title=ilike.*${encodeURIComponent(keyword)}*`;
  }

  const res = await fetch(
    `${SB_URL}/rest/v1/ext_tdnet_disclosures?disclosed_at=gte.${sinceStr}${codeFilter}${titleFilter}&select=code,disclosed_at,title,category&order=disclosed_at.desc&limit=30`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "適時開示データの取得に失敗しました。";
  const disclosures = await res.json();

  const CATALYST_KEYWORDS = ["上方修正", "増配", "自社株買", "株式分割", "特別配当", "業績予想の修正", "復配", "株主優待"];

  const lines: string[] = [];
  const scopeStr = scope === "watchlist" ? "ウォッチリスト" : "全銘柄";
  lines.push(`📰 ${scopeStr}の適時開示（直近${days}日）${keyword ? ` [${keyword}]` : ""}\n`);

  if (disclosures.length > 0) {
    for (const d of disclosures) {
      const date = d.disclosed_at?.slice(0, 10) ?? "";
      const isCatalyst = CATALYST_KEYWORDS.some((kw) => d.title.includes(kw));
      const mark = isCatalyst ? "🔥" : "📄";
      lines.push(`${mark} ${d.code} ${date}\n   ${d.title}`);
    }

    const catalystCount = disclosures.filter((d: { title: string }) =>
      CATALYST_KEYWORDS.some((kw) => d.title.includes(kw)),
    ).length;
    if (catalystCount > 0) {
      lines.push(`\n🔥 = カタリスト候補（上方修正・増配・自社株買い等）: ${catalystCount}件`);
    }
  } else {
    const scopeLabel = scope === "watchlist" ? "ウォッチリスト銘柄の" : "";
    const kwStr = keyword ? `「${keyword}」を含む` : "";
    lines.push(`直近${days}日間で${scopeLabel}${kwStr}適時開示はありません。`);
  }

  // EDINET大量保有報告を追加取得
  let holdingFilter = "";
  if (scope === "watchlist" && watchCodes.length > 0) {
    holdingFilter = `&issuer_code=in.(${watchCodes.join(",")})`;
  }
  const holdRes = await fetch(
    `${SB_URL}/rest/v1/edinet_large_holdings?disc_date=gte.${sinceStr}${holdingFilter}` +
      `&select=issuer_code,issuer_name,filer_name,holding_ratio,disc_date,doc_description` +
      `&order=disc_date.desc,submit_date.desc&limit=50`,
    { headers: sbHeaders() },
  );
  if (holdRes.ok) {
    let holdings: Record<string, unknown>[] = await holdRes.json();
    // 自己申告（提出者≒対象企業）は除外。売却は除外せず方向性を表示する
    holdings = holdings.filter(
      (h) => !isSelfFiling(h.filer_name as string, h.issuer_name as string),
    );
    // 過半数超（51%以上）はスクイーズアウト対象で上値が見込めないため除外
    holdings = holdings.filter(
      (h) => h.holding_ratio == null || Math.abs(h.holding_ratio as number) < MAJORITY_HOLDING_THRESHOLD,
    );

    // 銘柄+提出者ごとに、期間内で最も古い/新しい保有比率を集計。
    // 同一提出者の開示が複数あれば「5.2%→10.1%」のように変化を見せる
    const ratioHistory = new Map<string, { oldest: number; newest: number }>();
    for (const h of holdings) {
      if (h.holding_ratio == null) continue;
      const key = `${h.issuer_code}::${h.filer_name}`;
      const ratio = h.holding_ratio as number;
      const existing = ratioHistory.get(key);
      if (!existing) {
        ratioHistory.set(key, { oldest: ratio, newest: ratio });
      } else {
        // holdingsはdisc_date.desc順なので、後から見つかるものほど古い
        existing.oldest = ratio;
      }
    }

    // 同一銘柄の訂正報告等の重複を除き、銘柄ごとに最新1件だけ残す
    const seenCodes = new Set<string>();
    const deduped: Record<string, unknown>[] = [];
    for (const h of holdings) {
      const code = h.issuer_code as string;
      if (seenCodes.has(code)) continue;
      seenCodes.add(code);
      deduped.push(h);
    }
    // 法人/ファンドを優先、個人名の提出者は後回し。その中で保有比率が大きい順
    deduped.sort((a, b) => {
      const aInd = isIndividualFiler(a.filer_name as string) ? 1 : 0;
      const bInd = isIndividualFiler(b.filer_name as string) ? 1 : 0;
      if (aInd !== bInd) return aInd - bInd;
      return Math.abs((b.holding_ratio as number) ?? 0) - Math.abs((a.holding_ratio as number) ?? 0);
    });

    if (deduped.length > 0) {
      lines.push(`\n🏦 大量保有報告（直近${days}日・銘柄ごとに最新1件・全${deduped.length}銘柄）\n`);
      for (const h of deduped) {
        let ratio: string;
        if (h.holding_ratio == null) {
          ratio = "不明";
        } else {
          const hist = ratioHistory.get(`${h.issuer_code}::${h.filer_name}`);
          ratio = hist && hist.oldest !== hist.newest
            ? `${hist.oldest.toFixed(2)}%→${hist.newest.toFixed(2)}%`
            : `${(h.holding_ratio as number).toFixed(2)}%`;
        }
        const direction = isSellDisclosure(h.doc_description as string) ? "📉売り" : "📈買い";
        lines.push(`📋 ${h.issuer_code} ${h.issuer_name}\n   ${h.filer_name} 保有比率${ratio} ${direction} (${h.disc_date})`);
      }
    } else {
      lines.push(`\n🏦 大量保有報告（直近${days}日）: なし`);
    }
  }

  return lines.join("\n");
}

async function askClaude(userMessage: string, context: string, userId: string): Promise<string> {
  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

  const systemPrompt = `あなたは株アラートBot。友人に話すように、カジュアルだけど的確に答える。敬語は「です・ます」止め。

## データ
${context || "（本日のデータはまだありません）"}

## 判断基準
- 下落確率: <8%=🟢安全圏、8-15%=🟡普通、≥15%=🔴危険
- 下落モデルAUC 0.771で精度は高め。上昇モデルは精度が低いのでnetスコアは絶対に使わない
- 全銘柄の平均下落確率≥15% → 市場全体が危険、キャッシュ退避推奨

## 回答スタイル
- 結論を最初に言う。「買い」「様子見」「危険」等をはっきり
- ウォッチリスト全体への質問（「買い時は？」「安全なのは？」「下落確率教えて」等）→ 対象銘柄を1件ずつ箇条書きで列挙し、各行に必ず「株価・前日比・前週比・下落確率」を含める（例: 「🟡 8058 三菱商事: 4447円（前日比-1.8%, 前週比+0.9%） 下落確率14.4%」）。contextの各銘柄行に株価・前日比・前週比が入っているので省略せずそのまま使う。条件を満たさない銘柄も同じ形式で示し、末尾でまとめて評価する
- 銘柄照会では最初に現在の株価・前日比・前週比を表示し、直近の株価傾向（20日相対強度や出来高変化）にも触れる
- 基礎数値（株価・下落確率・PER・PBR・配当利回り・ROE等）を省略せず表示する。PER・PBR・ROE・配当利回りは必ず意味と目安を付けて表示する（例: 「PER（株価収益率＝株価÷EPS。低いほど割安）: 15.2（目安<15）」「PBR（株価純資産倍率＝株価÷BPS）: 1.3（目安<1）」「ROE（自己資本利益率＝純利益÷自己資本）: 12.5%（目安>10%）」「配当利回り（年間配当÷株価）: 3.2%（目安>3%）」）
- 数値には必ず指標名と単位をつける（✕「5日推移: 2.5→3.2」→ ○「下落確率の5日推移: 2.5%→3.2%」）
- 800文字以内に収める。冗長な解説は不要だが数値は削らない（🏦大量保有報告の全件列挙は例外。件数が多く800字を超えても、要約せず銘柄を省略しない）
- 「下落確率」と呼ぶ（dpとは言わない）
- 銘柄分析や売買判断に触れる回答の末尾には必ず「※本情報は参考情報であり、投資判断は自己責任でお願いします」と添える
- わからないことは「わかりません」と正直に言う。「サポートに問い合わせ」「提供者に確認」等の案内は絶対にしない（サポート窓口は存在しない）
- このBotが対応できるのは株・投資関連の相談のみ。対応外の質問には「株や投資に関することなら何でも聞いてください！」と返す

## 指標の目安
PER（株価収益率＝株価÷EPS。何年分の利益で元が取れるか。低いほど割安）<15=割安 / PBR（株価純資産倍率＝株価÷BPS。純資産の何倍で買われているか）<1=純資産割れ / ROE（自己資本利益率＝純利益÷自己資本。株主資本をどれだけ効率よく稼いでいるか）>10%=良好 / 配当利回り（年間配当÷株価）>3%=高配当 / Piotroski≥7=財務健全(9点満点) / 自己資本比率>40%=安定 / 営業利益率>10%=高収益 / 進捗率: 1Q>25%,2Q>50%,3Q>75%で順調

## 質問→使う指標
買い? → 下落確率+PER/PBR+ROE+配当 / 配当? → 利回り+配当性向+営業CF / 安全? → 下落確率+自己資本比率+Piotroski / 割安? → PER+PBR+BPS成長 / 成長? → 売上・利益成長率+進捗率 / ニュース? → TDnet適時開示 / 需給? → 空売り+信用残+出来高+需給シグナル / 比較? → セクター内順位+セクター平均PER

## 需給シグナル（自動生成）
lookup_stockの結果に【需給シグナル】セクションが含まれる。空売り急増→踏み上げ警戒、信用買い増→個人強気、貸借倍率高→将来の売り圧力等を自動判定。需給に触れる質問ではこのシグナルを活用して具体的に解説する

## セクター横断比較（自動生成）
lookup_stockの結果に【セクター比較】セクションが含まれる。同セクター内の安全性順位・PER比較・Top3銘柄を表示。「同業他社と比べてどう？」「セクターの中で割安？」等の質問で積極活用する。比較されていない場合も、セクター情報があれば言及する

## テーマ・業界・ビジネスモデルの質問
- 「ロボタクシー関連は？」「半導体関連銘柄」「○○のビジネスモデルは？」等のテーマ質問には、あなたの学習知識を活用して積極的に答える
- 企業の事業内容・業界ポジション・技術的強み・テーマとの関連性は、あなたが知っている範囲で具体的に説明する。「情報範囲外」「IR資料を確認して」等と逃げない
- テーマに関連する銘柄を挙げるときは、lookup_stockで数値データも取得して合わせて提示する
- 知識が古い・不確かな場合はその旨を添えつつ、知っている範囲で答える

## 銘柄スクリーナー（screen_stocks）
「高配当で安全な銘柄」「PBR1倍割れの割安株」等の条件検索にはscreen_stocksを使う。ユーザーの自然言語をフィルターパラメータに変換する:
- 安全/低リスク → max_drop_prob=8
- 割安 → max_per=15 and/or max_pbr=1.0
- 高配当 → min_div_yield=3.0
- 財務健全 → min_piotroski=7
- 高ROE → min_roe=10
条件を組み合わせて使う。結果が少なすぎたら条件を緩める。結果の銘柄について補足コメント（セクター特性・注意点）を添える

## カタリスト速報（check_catalyst）
「最近の決算サプライズは？」「上方修正した銘柄」「ウォッチ銘柄のニュースは？」「大量保有報告」「機関投資家の動き」等にはcheck_catalystを使う。「ウォッチ銘柄の」「保有株の」と明示された場合のみscope='watchlist'、それ以外は必ずscope='all'（市場全体）。「ほかにも」「他の銘柄は」等の追加質問はscope='all'で呼び直す。keyword='上方修正'等で絞込可能。🔥マーク付きはカタリスト候補（上方修正・増配・自社株買い等）。🏦マーク付きはEDINET大量保有報告（銘柄ごとに最新1件・保有比率順に整理済み。📈買い/📉売りの方向性つき）。
🏦セクションが返ってきたら、800文字ルールより列挙を優先する。件数が多くても「ほぼ全銘柄が買い」「VCが積極的」等の傾向だけ述べて終わらせるのは禁止。1件だけ選んで語らず、返ってきた銘柄を全件そのまま箇条書きで列挙する（銘柄コード・会社名・提出者・保有比率・方向性・開示日を省略しない）。0件の場合のみ「大口の動きはありません」等と答える

## ツール使い分け
- **特定の1銘柄を聞かれた場合**（「トヨタはどう？」「三菱商事を分析して」）→ lookup_stock で最新データ取得
- **ウォッチリスト全体の状況**（「買い時の銘柄はある？」「ウォッチの中で安全なのは？」）→ contextのデータをそのまま使う。lookup_stockは不要
- **条件スクリーニング**（「高配当で安全な銘柄」）→ screen_stocks
- **適時開示・カタリスト・大量保有**（「上方修正は？」「大量取得は？」「機関の動き」）→ check_catalyst
- ウォッチ操作は該当ツールを使う

## 銘柄コード（ツール用）
トヨタ=7203, ソニー=6758, 任天堂=7974, キーエンス=6861, 三菱UFJ=8306, 三井住友FG=8316, みずほ=8411, SBI=8473, ソフトバンクG=9984, NTT=9432, KDDI=9433, ファストリ=9983, 東京エレクトロン=8035, 信越化学=4063
※上記以外は名前の一部をqueryに渡す。DBの銘柄名は全角英数字なので注意。

## 機能案内（「何ができる？」と聞かれたら全部伝える）
1. 📊 毎日20時に日経225シグナル通知（投資継続OK/キャッシュ推奨）
2. 📋 ウォッチ銘柄の株価・下落確率・買い時/売り時を日次通知
3. 💬 銘柄分析・相場相談（会話の文脈を覚えている）
4. 📝 ウォッチリスト管理（「SBIをウォッチして」「リスト」等）
5. 📰 適時開示・空売り・信用残高も自動表示
6. 🔍 条件スクリーニング（「高配当で安全な銘柄」「PBR1倍割れ」等）
7. 🔥 決算サプライズ速報（上方修正・増配・自社株買いを自動検出）
8. 🏦 EDINET大量保有報告（5%超保有・機関投資家の動き）`;

  const history = await getRecentMessages(userId);
  const messages: Anthropic.MessageParam[] = [];

  let lastRole = "";
  for (const h of history) {
    if (h.role === lastRole) continue;
    messages.push({ role: h.role as "user" | "assistant", content: h.content });
    lastRole = h.role;
  }
  if (messages.length > 0 && messages[messages.length - 1].role === "user") {
    messages[messages.length - 1] = { role: "user", content: userMessage };
  } else {
    messages.push({ role: "user", content: userMessage });
  }

  const MAX_TURNS = 3;
  let lastText = "";

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    const response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 1024,
      system: systemPrompt,
      tools: TOOLS,
      messages,
    });

    const textBlocks = response.content.filter(
      (b): b is Anthropic.TextBlock => b.type === "text",
    );
    const toolBlocks = response.content.filter(
      (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
    );

    if (textBlocks.length > 0) {
      lastText = textBlocks.map((b) => b.text).join("\n");
    }

    if (toolBlocks.length === 0) {
      return lastText || "回答を生成できませんでした。";
    }

    const toolResults: Anthropic.ToolResultBlockParam[] = [];
    for (const tool of toolBlocks) {
      const result = await executeToolCall(tool.name, tool.input as Record<string, unknown>, userId);
      toolResults.push({ type: "tool_result", tool_use_id: tool.id, content: result });
    }

    messages.push({ role: "assistant", content: response.content });
    messages.push({ role: "user", content: toolResults });
  }

  if (lastText) return lastText;

  const finalResponse = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 600,
    system: systemPrompt,
    messages: [
      ...messages,
      { role: "user", content: "ここまでのツール結果をもとに、ユーザーの質問に日本語で簡潔に答えてください。" },
    ],
  });
  const finalText = finalResponse.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("\n");

  return finalText || "うまく回答できませんでした。もう一度お試しください。";
}

// ── Webhook エントリポイント ─────────────────

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { status: 200 });
  }

  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const rawBody = await req.text();
  const signature = req.headers.get("x-line-signature") ?? "";

  if (!(await verifySignature(rawBody, signature))) {
    return Response.json({ error: "Invalid signature" }, { status: 403 });
  }

  const body = JSON.parse(rawBody);
  const events = body.events ?? [];

  for (const event of events) {
    const userId = event.source?.userId as string;
    if (!userId) continue;

    if (event.type === "follow") {
      await getOrCreateUser(userId);
      await replyToLine(event.replyToken, "📈 stock-alert Bot へようこそ！\n\n「ヘルプ」と送ると使い方を確認できます。\n「ランキング」で本日の注目銘柄を表示します。");
      continue;
    }

    if (event.type !== "message" || event.message?.type !== "text") continue;

    const userMessage = event.message.text as string;
    const replyToken = event.replyToken as string;

    if (!checkRateLimit(userId)) {
      await replyToLine(replyToken, "⚠️ メッセージが多すぎます。1分ほどお待ちください。");
      continue;
    }

    try {
      const user = await getOrCreateUser(userId);
      await saveMessage(userId, "user", userMessage);

      const cmd = await handleCommand(userMessage, userId, user);
      if (cmd.handled) {
        await saveMessage(userId, "assistant", cmd.reply);
        await replyToLine(replyToken, cmd.reply);
        trimHistory(userId).catch(() => {});
        continue;
      }

      if (!isPremium(user)) {
        const aiLimit = checkAiLimit(userId);
        if (!aiLimit.allowed) {
          const reply = `⚠️ 本日のAI相談回数（${FREE_AI_LIMIT}回）を使い切りました。\n明日またご利用いただくか、プレミアムプランで無制限にできます。\n「プラン」で詳細確認`;
          await saveMessage(userId, "assistant", reply);
          await replyToLine(replyToken, reply);
          continue;
        }
      }

      await showLoadingAnimation(userId, 60);

      const context = await fetchMarketContext(userId, userMessage);
      let reply = await askClaude(userMessage, context, userId);
      if (!isPremium(user)) {
        recordAiUsage(userId);
        const totalQueries = await incrementTotalAiQueries(userId);
        if (totalQueries > 0 && totalQueries % UPGRADE_NUDGE_INTERVAL === 0) {
          reply += UPGRADE_NUDGE;
        }
      }
      await saveMessage(userId, "assistant", reply);
      await pushToLine(userId, reply);
      trimHistory(userId).catch(() => {});
    } catch (e) {
      console.error("[LINE webhook] Error:", e);
      try {
        await pushToLine(userId, "エラーが発生しました。しばらくしてからお試しください。");
      } catch {
        // push も失敗した場合は諦める
      }
    }
  }

  return Response.json({ ok: true });
});

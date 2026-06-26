import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import * as crypto from "crypto";

const LINE_CHANNEL_SECRET = process.env.LINE_CHANNEL_SECRET ?? "";
const LINE_CHANNEL_ACCESS_TOKEN = process.env.LINE_CHANNEL_ACCESS_TOKEN ?? "";
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY ?? "";

const SB_URL = (process.env.NEXT_PUBLIC_SUPABASE_URL ?? "").trim();
const SB_KEY = (process.env.SUPABASE_SERVICE_KEY ?? "").trim();

function sbHeaders(extra: Record<string, string> = {}) {
  return {
    apikey: SB_KEY,
    Authorization: `Bearer ${SB_KEY}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

function verifySignature(body: string, signature: string): boolean {
  if (!LINE_CHANNEL_SECRET) return false;
  const hash = crypto
    .createHmac("SHA256", LINE_CHANNEL_SECRET)
    .update(body)
    .digest("base64");
  return hash === signature;
}

async function replyToLine(replyToken: string, text: string): Promise<void> {
  const maxLen = 5000;
  const truncated =
    text.length > maxLen ? text.slice(0, maxLen - 3) + "..." : text;
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

// ── ウォッチリスト管理（ユーザー別） ────────────────────────

interface WatchItem {
  code: string;
  name: string;
  dp_threshold: number;
  line_user_id: string;
}

async function getWatchlist(userId: string): Promise<WatchItem[]> {
  const res = await fetch(
    `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold&order=created_at`,
    { headers: sbHeaders() }
  );
  return res.ok ? await res.json() : [];
}

async function addToWatchlist(
  userId: string,
  code: string,
  name: string,
  threshold: number
): Promise<boolean> {
  const res = await fetch(`${SB_URL}/rest/v1/dp_watchlist`, {
    method: "POST",
    headers: sbHeaders({ Prefer: "resolution=merge-duplicates" }),
    body: JSON.stringify({
      line_user_id: userId,
      code,
      name,
      dp_threshold: threshold,
    }),
  });
  return res.ok;
}

async function removeFromWatchlist(
  userId: string,
  code: string
): Promise<boolean> {
  const res = await fetch(
    `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&code=eq.${code}`,
    { method: "DELETE", headers: sbHeaders() }
  );
  return res.ok;
}

async function lookupStock(
  code: string
): Promise<{
  code: string;
  name: string;
  close: number;
  drop_prob: number;
} | null> {
  const res = await fetch(
    `${SB_URL}/rest/v1/gen_rankings?code=eq.${code}&select=code,name,close,drop_prob&order=date.desc&limit=1`,
    { headers: sbHeaders() }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return rows.length > 0 ? rows[0] : null;
}

async function searchStockByName(
  keyword: string
): Promise<{ code: string; name: string }[]> {
  const res = await fetch(
    `${SB_URL}/rest/v1/gen_stock_meta?name=ilike.*${encodeURIComponent(keyword)}*&select=code,name&limit=5`,
    { headers: sbHeaders() }
  );
  return res.ok ? await res.json() : [];
}

// ── コマンド処理 ─────────────────────────────────────────

type CommandResult = { handled: true; reply: string } | { handled: false };

async function handleCommand(
  text: string,
  userId: string
): Promise<CommandResult> {
  const trimmed = text.trim();

  // ウォッチリスト一覧
  if (/^(ウォッチ|リスト|一覧)$/i.test(trimmed)) {
    const list = await getWatchlist(userId);
    if (list.length === 0) {
      return {
        handled: true,
        reply:
          "ウォッチリストは空です。\n「ウォッチ 8473」で追加できます。",
      };
    }
    const lines = ["📋 ウォッチリスト:"];
    for (const item of list) {
      const stock = await lookupStock(item.code);
      const dpStr = stock ? `dp=${stock.drop_prob}%` : "dp=--";
      const priceStr = stock ? `${stock.close.toLocaleString()}円` : "";
      const status =
        stock && stock.drop_prob < item.dp_threshold ? "🔔買い時！" : "";
      lines.push(
        `  ${item.code} ${item.name} (閾値dp<${item.dp_threshold}) ${dpStr} ${priceStr} ${status}`
      );
    }
    lines.push("\n「ウォッチ 銘柄コード」で追加");
    lines.push("「解除 銘柄コード」で削除");
    return { handled: true, reply: lines.join("\n") };
  }

  // ウォッチ追加: 「ウォッチ 8473」「ウォッチ 8473 10」「ウォッチ SBI」
  const addMatch = trimmed.match(
    /^ウォッチ\s+(.+?)(?:\s+(\d+(?:\.\d+)?))?$/i
  );
  if (addMatch) {
    const target = addMatch[1].trim();
    const threshold = addMatch[2] ? parseFloat(addMatch[2]) : 8.0;

    if (/^\d{4}$/.test(target)) {
      const stock = await lookupStock(target);
      if (!stock) {
        return {
          handled: true,
          reply: `${target} はランキングに見つかりません。銘柄コード4桁で指定してください。`,
        };
      }
      await addToWatchlist(userId, stock.code, stock.name, threshold);
      return {
        handled: true,
        reply:
          `✅ ${stock.name}(${stock.code}) をウォッチリストに追加\n` +
          `閾値: dp < ${threshold}\n` +
          `現在のdp: ${stock.drop_prob}%\n` +
          `株価: ${stock.close.toLocaleString()}円`,
      };
    }

    const matches = await searchStockByName(target);
    if (matches.length === 0) {
      return {
        handled: true,
        reply: `「${target}」に該当する銘柄が見つかりません。コード4桁で指定してください。`,
      };
    }
    if (matches.length === 1) {
      const stock = await lookupStock(matches[0].code);
      const name = stock?.name ?? matches[0].name;
      await addToWatchlist(userId, matches[0].code, name, threshold);
      return {
        handled: true,
        reply:
          `✅ ${name}(${matches[0].code}) をウォッチリストに追加\n` +
          `閾値: dp < ${threshold}\n` +
          (stock
            ? `現在のdp: ${stock.drop_prob}%\n株価: ${stock.close.toLocaleString()}円`
            : ""),
      };
    }
    const options = matches.map((m) => `  ${m.code} ${m.name}`).join("\n");
    return {
      handled: true,
      reply: `「${target}」に複数該当:\n${options}\n\nコードで指定してください。\n例: ウォッチ ${matches[0].code}`,
    };
  }

  // ウォッチ解除: 「解除 8473」
  const removeMatch = trimmed.match(/^(解除|削除|外す)\s+(\d{4})$/i);
  if (removeMatch) {
    const code = removeMatch[2];
    await removeFromWatchlist(userId, code);
    return {
      handled: true,
      reply: `🗑 ${code} をウォッチリストから削除しました。`,
    };
  }

  // 銘柄照会: 「8473」（4桁数字のみ）
  if (/^\d{4}$/.test(trimmed)) {
    const stock = await lookupStock(trimmed);
    if (!stock) {
      return { handled: true, reply: `${trimmed} のデータが見つかりません。` };
    }
    const dpStatus =
      stock.drop_prob < 8
        ? "🟢安全圏"
        : stock.drop_prob >= 15
          ? "🔴危険"
          : "🟡通常";
    return {
      handled: true,
      reply:
        `📊 ${stock.name}(${stock.code})\n` +
        `株価: ${stock.close.toLocaleString()}円\n` +
        `dp: ${stock.drop_prob}% ${dpStatus}\n\n` +
        `「ウォッチ ${stock.code}」でウォッチに追加`,
    };
  }

  return { handled: false };
}

// ── 市場データ取得 ───────────────────────────────────────

async function fetchMarketContext(userId: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const today = new Date().toISOString().slice(0, 10);

  const [rankRes, n225Res, watchRes] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob,rise_prob,net,recommend&order=net.desc&limit=20`,
      { headers: sbHeaders() }
    ),
    fetch(
      `${SB_URL}/rest/v1/yahoo_market_index?ticker=eq.N225&order=date.desc&limit=1&select=date,close`,
      { headers: sbHeaders() }
    ),
    fetch(
      `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold`,
      { headers: sbHeaders() }
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
      .filter((r: any) => r.drop_prob != null)
      .map((r: any) => r.drop_prob);
    const avgDp =
      dps.length > 0
        ? dps.reduce((a: number, b: number) => a + b, 0) / dps.length
        : null;

    if (avgDp != null) {
      const signal = avgDp >= 15 ? "🔴キャッシュ推奨" : "🟢投資継続OK";
      lines.push(`市場平均dp: ${avgDp.toFixed(1)}% → ${signal}`);
    }

    lines.push(`\n本日のトップ10:`);
    for (const r of rankings.slice(0, 10)) {
      lines.push(
        `  ${r.code} ${r.name}: net=${r.net}% dp=${r.drop_prob}% ${r.recommend ?? ""}`
      );
    }
  }

  if (watchlist.length > 0) {
    lines.push(`\nこのユーザーのウォッチリスト:`);
    for (const w of watchlist) {
      const stock = rankings.find((r: any) => r.code === w.code);
      const dpStr = stock ? `dp=${stock.drop_prob}%` : "dp=--";
      lines.push(
        `  ${w.code} ${w.name}: ${dpStr} (閾値dp<${w.dp_threshold})`
      );
    }
  }

  return lines.join("\n");
}

// ── Claude API ──────────────────────────────────────────

async function askClaude(
  userMessage: string,
  context: string
): Promise<string> {
  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

  const systemPrompt = `あなたは日本株投資のアシスタントBot「stock-alert」です。
ユーザーの質問に簡潔に日本語で答えてください。

以下はリアルタイムの市場データです:
${context || "（本日のデータはまだありません）"}

あなたが知っていること:
- XGBoostで63日先の±15%変動を予測するモデルを使用
- drop_prob(dp): 下落確率。dp<8は安全圏、dp≥15は危険
- 全銘柄の平均dp≥15なら日経ETFをキャッシュに退避すべき
- ウォッチリストの銘柄はdpが閾値を下回ったら買いタイミング
- メガバンク(8306三菱UFJ/8316三井住友FG/8411みずほ)は金利上昇で構造的に強い

コマンドの案内（ユーザーがやり方を聞いたら教える）:
- 「ウォッチ 8473」→ ウォッチリストに追加（dp<8で通知）
- 「ウォッチ 8473 10」→ 閾値dp<10で追加
- 「ウォッチ SBI」→ 名前で検索して追加
- 「解除 8473」→ ウォッチリストから削除
- 「リスト」→ ウォッチリスト一覧
- 「8473」→ 銘柄の現在状況を表示

ルール:
- 投資は自己責任である旨を必要に応じて添える
- 500文字以内で回答する
- データがない質問には正直に「わからない」と答える`;

  const msg = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 1024,
    system: systemPrompt,
    messages: [{ role: "user", content: userMessage }],
  });

  return msg.content[0].type === "text"
    ? msg.content[0].text
    : "回答を生成できませんでした。";
}

// ── Webhook エントリポイント ─────────────────────────────

export async function POST(req: NextRequest) {
  const rawBody = await req.text();
  const signature = req.headers.get("x-line-signature") ?? "";

  if (LINE_CHANNEL_SECRET && !verifySignature(rawBody, signature)) {
    return NextResponse.json({ error: "Invalid signature" }, { status: 403 });
  }

  const body = JSON.parse(rawBody);
  const events = body.events ?? [];

  for (const event of events) {
    if (event.type !== "message" || event.message?.type !== "text") continue;

    const userMessage = event.message.text as string;
    const replyToken = event.replyToken as string;
    const userId = event.source?.userId as string;

    if (!userId) continue;

    try {
      const cmd = await handleCommand(userMessage, userId);
      if (cmd.handled) {
        await replyToLine(replyToken, cmd.reply);
        continue;
      }

      const context = await fetchMarketContext(userId);
      const reply = await askClaude(userMessage, context);
      await replyToLine(replyToken, reply);
    } catch (e: any) {
      console.error("[LINE webhook] Error:", e);
      await replyToLine(
        replyToken,
        "エラーが発生しました。しばらくしてからお試しください。"
      );
    }
  }

  return NextResponse.json({ ok: true });
}

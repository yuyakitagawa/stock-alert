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
  dp_sell_threshold: number;
  line_user_id: string;
}

async function getWatchlist(userId: string): Promise<WatchItem[]> {
  const res = await fetch(
    `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold,dp_sell_threshold&order=created_at`,
    { headers: sbHeaders() }
  );
  return res.ok ? await res.json() : [];
}

async function addToWatchlist(
  userId: string,
  code: string,
  name: string,
  threshold: number,
  sellThreshold: number = 20.0
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
      let status = "";
      if (stock) {
        if (stock.drop_prob < item.dp_threshold) status = "🔔買い時！";
        else if (stock.drop_prob >= (item.dp_sell_threshold ?? 20)) status = "⚠️売り検討";
      }
      lines.push(
        `  ${item.code} ${item.name}\n  買<${item.dp_threshold} 売≥${item.dp_sell_threshold ?? 20} ${dpStr} ${priceStr} ${status}`
      );
    }
    lines.push("\n「ウォッチ 銘柄コード」で追加");
    lines.push("「解除 銘柄コード」で削除");
    return { handled: true, reply: lines.join("\n") };
  }

  // ウォッチ追加: 「ウォッチ 8473」「ウォッチ 8473 10」「ウォッチ 8473 10 20」「ウォッチ SBI」
  const addMatch = trimmed.match(
    /^ウォッチ\s+(.+?)(?:\s+(\d+(?:\.\d+)?))?(?:\s+(\d+(?:\.\d+)?))?$/i
  );
  if (addMatch) {
    const target = addMatch[1].trim();
    const threshold = addMatch[2] ? parseFloat(addMatch[2]) : 8.0;
    const sellThreshold = addMatch[3] ? parseFloat(addMatch[3]) : 20.0;

    if (/^\d{4}$/.test(target)) {
      const stock = await lookupStock(target);
      if (!stock) {
        return {
          handled: true,
          reply: `${target} はランキングに見つかりません。銘柄コード4桁で指定してください。`,
        };
      }
      await addToWatchlist(userId, stock.code, stock.name, threshold, sellThreshold);
      return {
        handled: true,
        reply:
          `✅ ${stock.name}(${stock.code}) をウォッチリストに追加\n` +
          `買い閾値: dp < ${threshold}\n` +
          `売り閾値: dp ≥ ${sellThreshold}\n` +
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
      await addToWatchlist(userId, matches[0].code, name, threshold, sellThreshold);
      return {
        handled: true,
        reply:
          `✅ ${name}(${matches[0].code}) をウォッチリストに追加\n` +
          `買い閾値: dp < ${threshold}\n` +
          `売り閾値: dp ≥ ${sellThreshold}\n` +
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

async function fetchStockDetail(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const [rankRes, finRes] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?code=eq.${code}&select=code,name,close,drop_prob,rise_prob,net,recommend,per,pbr,piotroski,bps_growth,eps_surprise,vol,rel20&order=date.desc&limit=5`,
      { headers: sbHeaders() }
    ),
    fetch(
      `${SB_URL}/rest/v1/jquants_fin_summary?code=eq.${code}&select=code,disc_date,eps,bps,div_ann,payout_ratio&order=disc_date.desc&limit=1`,
      { headers: sbHeaders() }
    ),
  ]);

  const ranks = rankRes.ok ? await rankRes.json() : [];
  const fins = finRes.ok ? await finRes.json() : [];
  const lines: string[] = [];

  if (ranks.length > 0) {
    const r = ranks[0];
    lines.push(`【${r.name}(${r.code})の詳細データ】`);
    lines.push(`株価: ${r.close}円, dp: ${r.drop_prob}%, net: ${r.net}%`);
    lines.push(`PER: ${r.per}, PBR: ${r.pbr}, Piotroski: ${r.piotroski}`);
    lines.push(`BPS成長: ${r.bps_growth}, EPSサプライズ: ${r.eps_surprise}`);
    lines.push(`出来高: ${r.vol}, 20日相対強度: ${r.rel20}`);
    lines.push(`推奨: ${r.recommend ?? "なし"}`);
    if (ranks.length > 1) {
      lines.push(`過去5日のdp推移: ${ranks.map((x: any) => x.drop_prob).join(" → ")}`);
    }
  }
  if (fins.length > 0) {
    const f = fins[0];
    lines.push(`\n【決算データ(${f.disc_date})】`);
    lines.push(`EPS: ${f.eps}, BPS: ${f.bps}, 年間配当: ${f.div_ann}, 配当性向: ${f.payout_ratio}`);
  }

  return lines.join("\n");
}

async function fetchMarketContext(userId: string, userMessage: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const today = new Date().toISOString().slice(0, 10);

  const [rankRes, n225Res, watchRes] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob,rise_prob,net,recommend,per,pbr&order=net.desc&limit=20`,
      { headers: sbHeaders() }
    ),
    fetch(
      `${SB_URL}/rest/v1/yahoo_market_index?ticker=eq.N225&order=date.desc&limit=1&select=date,close`,
      { headers: sbHeaders() }
    ),
    fetch(
      `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold,dp_sell_threshold`,
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
        `  ${r.code} ${r.name}: net=${r.net}% dp=${r.drop_prob}% PER=${r.per} PBR=${r.pbr} ${r.recommend ?? ""}`
      );
    }
  }

  if (watchlist.length > 0) {
    lines.push(`\nこのユーザーのウォッチリスト:`);
    for (const w of watchlist) {
      const stock = rankings.find((r: any) => r.code === w.code);
      const dpStr = stock ? `dp=${stock.drop_prob}%` : "dp=--";
      lines.push(
        `  ${w.code} ${w.name}: ${dpStr} (買<${w.dp_threshold} 売≥${w.dp_sell_threshold ?? 20})`
      );
    }
  }

  // ユーザーが特定銘柄について質問している場合、詳細データを追加取得
  const codeMatch = userMessage.match(/(\d{4})/);
  if (codeMatch) {
    const detail = await fetchStockDetail(codeMatch[1]);
    if (detail) lines.push("\n" + detail);
  }

  return lines.join("\n");
}

// ── Claude API (tool use で自然言語からウォッチ操作) ─────

const TOOLS: Anthropic.Tool[] = [
  {
    name: "add_watchlist",
    description:
      "ウォッチリストに銘柄を追加する。ユーザーが「〇〇をウォッチして」「〇〇を監視したい」「〇〇が下がったら教えて」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "銘柄コード(4桁)または銘柄名の一部。例: '8473', 'SBI', '三菱UFJ'",
        },
        buy_threshold: {
          type: "number",
          description: "買い閾値(dp%)。指定なければ8.0",
        },
        sell_threshold: {
          type: "number",
          description: "売り閾値(dp%)。指定なければ20.0",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "remove_watchlist",
    description:
      "ウォッチリストから銘柄を削除する。「〇〇を外して」「〇〇の監視やめて」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "銘柄コード(4桁)または銘柄名の一部",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "show_watchlist",
    description:
      "ウォッチリストの一覧を表示する。「ウォッチリスト見せて」「今何を監視してる？」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {},
    },
  },
  {
    name: "lookup_stock",
    description:
      "特定銘柄の現在の詳細情報を調べる。「〇〇の状況は？」「〇〇って今どう？」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "銘柄コード(4桁)または銘柄名の一部",
        },
      },
      required: ["query"],
    },
  },
];

async function resolveCode(
  query: string
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
  input: Record<string, any>,
  userId: string
): Promise<string> {
  if (toolName === "show_watchlist") {
    const list = await getWatchlist(userId);
    if (list.length === 0) return "ウォッチリストは空です。銘柄名やコードを言ってもらえれば追加します。";
    const lines = ["📋 ウォッチリスト:"];
    for (const item of list) {
      const stock = await lookupStock(item.code);
      const dpStr = stock ? `dp=${stock.drop_prob}%` : "dp=--";
      const priceStr = stock ? `${stock.close.toLocaleString()}円` : "";
      let status = "";
      if (stock) {
        if (stock.drop_prob < item.dp_threshold) status = "🔔買い時！";
        else if (stock.drop_prob >= (item.dp_sell_threshold ?? 20)) status = "⚠️売り検討";
      }
      lines.push(`  ${item.code} ${item.name}\n  買<${item.dp_threshold} 売≥${item.dp_sell_threshold ?? 20} ${dpStr} ${priceStr} ${status}`);
    }
    return lines.join("\n");
  }

  if (toolName === "add_watchlist") {
    const resolved = await resolveCode(input.query);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    const buyTh = input.buy_threshold ?? 8.0;
    const sellTh = input.sell_threshold ?? 20.0;
    await addToWatchlist(userId, resolved.code, resolved.name, buyTh, sellTh);
    const stock = await lookupStock(resolved.code);
    return `✅ ${resolved.name}(${resolved.code}) をウォッチリストに追加\n買い閾値: dp < ${buyTh} / 売り閾値: dp ≥ ${sellTh}` +
      (stock ? `\n現在のdp: ${stock.drop_prob}% / 株価: ${stock.close.toLocaleString()}円` : "");
  }

  if (toolName === "remove_watchlist") {
    const resolved = await resolveCode(input.query);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    await removeFromWatchlist(userId, resolved.code);
    return `🗑 ${resolved.name}(${resolved.code}) をウォッチリストから削除しました。`;
  }

  if (toolName === "lookup_stock") {
    const resolved = await resolveCode(input.query);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    const detail = await fetchStockDetail(resolved.code);
    return detail || `${resolved.code} のデータが見つかりません。`;
  }

  return "不明な操作です。";
}

async function askClaude(
  userMessage: string,
  context: string,
  userId: string
): Promise<string> {
  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

  const systemPrompt = `あなたは日本株投資の専門アシスタントBot「stock-alert」です。
ユーザーの株に関する相談に、実際のデータに基づいて日本語で答えてください。

以下はリアルタイムの市場データです:
${context || "（本日のデータはまだありません）"}

あなたの分析フレームワーク:
- XGBoostで63日先の±15%変動を予測。下落モデル(AUC 0.766)の精度が高い
- drop_prob(dp): 下落確率。dp<8は安全圏、dp≥15は危険水準
- net = rise_prob - drop_prob。高いほど上昇期待
- 全銘柄の平均dp≥15なら日経ETFをキャッシュに退避すべき（マーケットタイミング）
- PBR<1は割安、PER<15は割安圏、Piotroski≥7は財務健全
- ウォッチリストの銘柄はdpが買い閾値を下回ったら買い、売り閾値以上なら売り検討

株の相談の答え方:
- データにある指標（dp, PER, PBR, Piotroski, BPS成長, EPSサプライズ等）を使って具体的に分析する
- 「この株は買いか？」→ dpの水準、ファンダメンタルズ、市場環境を総合的に判断
- セクター動向や同業比較も可能な範囲で言及する
- ユーザーが銘柄の監視や通知を求めたら、toolを使ってウォッチリストを操作する
- 銘柄の詳細を聞かれたら lookup_stock ツールで最新データを取得してから回答する

ルール:
- 投資は自己責任である旨を必要に応じて添える
- 800文字以内で回答する
- データがない質問には正直に「わからない」と答える`;

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
  ];

  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: systemPrompt,
    tools: TOOLS,
    messages,
  });

  // tool_use がなければテキストをそのまま返す
  const textBlocks = response.content.filter(
    (b): b is Anthropic.TextBlock => b.type === "text"
  );
  const toolBlocks = response.content.filter(
    (b): b is Anthropic.ToolUseBlock => b.type === "tool_use"
  );

  if (toolBlocks.length === 0) {
    return textBlocks.map((b) => b.text).join("\n") || "回答を生成できませんでした。";
  }

  // tool を実行して結果を返す
  const toolResults: Anthropic.ToolResultBlockParam[] = [];
  for (const tool of toolBlocks) {
    const result = await executeToolCall(
      tool.name,
      tool.input as Record<string, any>,
      userId
    );
    toolResults.push({
      type: "tool_result",
      tool_use_id: tool.id,
      content: result,
    });
  }

  // tool結果を含めて再度Claudeに投げ、最終回答を得る
  messages.push({ role: "assistant", content: response.content });
  messages.push({ role: "user", content: toolResults });

  const finalResponse = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: systemPrompt,
    tools: TOOLS,
    messages,
  });

  const finalText = finalResponse.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("\n");

  return finalText || "操作が完了しました。";
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

      const context = await fetchMarketContext(userId, userMessage);
      const reply = await askClaude(userMessage, context, userId);
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

import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import * as crypto from "crypto";

const LINE_CHANNEL_SECRET = process.env.LINE_CHANNEL_SECRET ?? "";
const LINE_CHANNEL_ACCESS_TOKEN = process.env.LINE_CHANNEL_ACCESS_TOKEN ?? "";
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY ?? "";

const SB_URL = (process.env.NEXT_PUBLIC_SUPABASE_URL ?? "").trim();
const SB_KEY = (process.env.SUPABASE_SERVICE_KEY ?? "").trim();

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

async function fetchMarketContext(): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";
  const headers = { apikey: SB_KEY, Authorization: `Bearer ${SB_KEY}` };

  const today = new Date().toISOString().slice(0, 10);

  const [rankRes, n225Res] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob,rise_prob,net,recommend&order=net.desc&limit=20`,
      { headers }
    ),
    fetch(
      `${SB_URL}/rest/v1/yahoo_market_index?ticker=eq.N225&order=date.desc&limit=1&select=date,close`,
      { headers }
    ),
  ]);

  const rankings = rankRes.ok ? await rankRes.json() : [];
  const n225 = n225Res.ok ? await n225Res.json() : [];

  const lines: string[] = [];

  if (n225.length > 0) {
    lines.push(`N225: ${n225[0].close}円 (${n225[0].date})`);
  }

  if (rankings.length > 0) {
    const dps = rankings
      .filter((r: any) => r.drop_prob != null)
      .map((r: any) => r.drop_prob);
    const avgDp = dps.length > 0 ? dps.reduce((a: number, b: number) => a + b, 0) / dps.length : null;

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

    const sbi = rankings.find((r: any) => r.code === "8473");
    if (sbi) {
      lines.push(`\nSBI HD(8473): dp=${sbi.drop_prob}% 株価=${sbi.close}円`);
    }
  }

  return lines.join("\n");
}

async function askClaude(userMessage: string, context: string): Promise<string> {
  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

  const systemPrompt = `あなたは日本株投資のアシスタントBot「stock-alert」です。
ユーザーの質問に簡潔に日本語で答えてください。

以下はリアルタイムの市場データです:
${context || "（本日のデータはまだありません）"}

あなたが知っていること:
- XGBoostで63日先の±15%変動を予測するモデルを使用
- drop_prob(dp): 下落確率。dp<8は安全圏、dp≥15は危険
- 全銘柄の平均dp≥15なら日経ETFをキャッシュに退避すべき
- SBI HD(8473)はdp<8になったら買いタイミング
- メガバンク(8306三菱UFJ/8316三井住友FG/8411みずほ)は金利上昇で構造的に強い

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

  return msg.content[0].type === "text" ? msg.content[0].text : "回答を生成できませんでした。";
}

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

    try {
      const context = await fetchMarketContext();
      const reply = await askClaude(userMessage, context);
      await replyToLine(replyToken, reply);
    } catch (e: any) {
      console.error("[LINE webhook] Error:", e);
      await replyToLine(replyToken, "エラーが発生しました。しばらくしてからお試しください。");
    }
  }

  return NextResponse.json({ ok: true });
}

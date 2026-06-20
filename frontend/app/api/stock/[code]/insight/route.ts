import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { anonHeaders, sbUrl } from "@/lib/supabase";
import type { Insight } from "@/lib/types";

// フェーズ1: 事業概要(拡張)・主要取引先・カタリスト評価・リスクの定性解説をClaudeで生成。
// 「利益の質(化粧/本業シュリンク)」は営業利益が必要だが現状Web側に無いためフェーズ2(別途データ基盤改修)で対応。
const MODEL_VER = "company-insight-v1";
const CACHE_DATE = "1970-01-02"; // description(v1)の 1970-01-01 と衝突させない

const serviceHeaders = () => ({
  apikey:        process.env.SUPABASE_SERVICE_KEY ?? "",
  Authorization: `Bearer ${process.env.SUPABASE_SERVICE_KEY ?? ""}`,
  "Content-Type": "application/json",
  Prefer:        "resolution=merge-duplicates",
});

async function getCached(code: string): Promise<Insight | null> {
  const res = await fetch(
    sbUrl(`gen_ai_analyses?code=eq.${code}&model_version=eq.${MODEL_VER}&select=summary&limit=1`),
    { headers: anonHeaders(), next: { revalidate: 86400 } }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  const raw = rows[0]?.summary as string | undefined;
  if (!raw) return null;
  try { return JSON.parse(raw) as Insight; } catch { return null; }
}

async function saveCache(code: string, insight: Insight): Promise<void> {
  const serviceKey = process.env.SUPABASE_SERVICE_KEY;
  if (!serviceKey) return;
  await fetch(`${process.env.NEXT_PUBLIC_SUPABASE_URL}/rest/v1/gen_ai_analyses`, {
    method: "POST",
    headers: serviceHeaders(),
    body: JSON.stringify([{
      code,
      date:          CACHE_DATE,
      summary:       JSON.stringify(insight),
      bull_points:   [],
      bear_points:   [],
      model_version: MODEL_VER,
    }]),
  });
}

function fmtNum(v: string | null, suffix = ""): string {
  if (v == null || v === "") return "不明";
  const n = Number(v);
  return Number.isFinite(n) ? `${n}${suffix}` : "不明";
}

export async function GET(
  req: Request,
  { params }: { params: Promise<{ code: string }> }
) {
  const { code } = await params;
  const url = new URL(req.url);
  const name   = url.searchParams.get("name")   ?? code;
  const sector = url.searchParams.get("sector") ?? "";
  const pbr    = url.searchParams.get("pbr");
  const per    = url.searchParams.get("per");
  const net    = url.searchParams.get("net");

  const cached = await getCached(code);
  if (cached) return NextResponse.json({ insight: cached });

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return NextResponse.json({ insight: null });

  const client = new Anthropic({ apiKey });

  const prompt = `あなたは日本株の中立的なアナリストです。上場企業「${name}」(コード ${code}、セクター ${sector || "不明"})について、投資家向けに簡潔な定性解説を作成してください。
参考指標(判明分): PBR ${fmtNum(pbr, "倍")} / PER ${fmtNum(per, "倍")} / ネットスコア(上昇-下落) ${fmtNum(net, "%")}。

次のJSONのみを出力(前後に文章不要):
{
  "business": "事業内容・強み・業界での立ち位置を120〜180字で。",
  "customers": "主要な取引先・販売先や顧客集中の傾向を80〜140字で。具体的な社名が確実でない場合は断定せず『〜とみられる』『一般に〜が多い』と表現し、顧客構造(例: 特定業種への依存)を述べる。",
  "catalyst": "上で与えたPBR等に基づき、割安度・株主還元やバリュー是正の余地を80〜140字で評価。数値が不明な項目は触れない。",
  "risks": ["リスクを2〜4個、各40字以内の配列で。"]
}
注意: 不確かな固有名詞・数値を作らないこと。わからない点は一般論で述べること。`;

  try {
    const msg = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 900,
      messages: [{ role: "user", content: prompt }],
    });
    const text = (msg.content[0] as { type: string; text: string }).text?.trim() ?? "";
    const jsonStr = text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
    const insight = JSON.parse(jsonStr) as Insight;
    if (!insight.business) return NextResponse.json({ insight: null });
    if (!Array.isArray(insight.risks)) insight.risks = [];
    await saveCache(code, insight);
    return NextResponse.json({ insight });
  } catch {
    return NextResponse.json({ insight: null });
  }
}

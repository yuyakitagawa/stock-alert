import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { anonHeaders, sbUrl } from "@/lib/supabase";

const MODEL_VER = "company-desc-v1";
const CACHE_DATE = "1970-01-01";

const serviceHeaders = () => ({
  apikey:        process.env.SUPABASE_SERVICE_KEY ?? "",
  Authorization: `Bearer ${process.env.SUPABASE_SERVICE_KEY ?? ""}`,
  "Content-Type": "application/json",
  Prefer:        "resolution=merge-duplicates",
});

async function getCached(code: string): Promise<string | null> {
  const res = await fetch(
    sbUrl(`ai_analyses?code=eq.${code}&model_version=eq.${MODEL_VER}&select=summary&limit=1`),
    { headers: anonHeaders(), next: { revalidate: 86400 } }
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return (rows[0]?.summary as string) ?? null;
}

async function saveCache(code: string, description: string): Promise<void> {
  const serviceKey = process.env.SUPABASE_SERVICE_KEY;
  if (!serviceKey) return;
  await fetch(`${process.env.NEXT_PUBLIC_SUPABASE_URL}/rest/v1/ai_analyses`, {
    method: "POST",
    headers: serviceHeaders(),
    body: JSON.stringify([{
      code,
      date:          CACHE_DATE,
      summary:       description,
      bull_points:   [],
      bear_points:   [],
      model_version: MODEL_VER,
    }]),
  });
}

async function getYahooDescription(code: string): Promise<string | null> {
  const UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";
  try {
    const r1 = await fetch("https://fc.yahoo.com", {
      headers: { "User-Agent": UA }, redirect: "follow", cache: "no-store",
    });
    const raw = r1.headers.get("set-cookie") ?? "";
    const cookies = raw.split(/,(?=[^ ][^,]+=)/).map(c => c.trim().split(";")[0]).filter(c => c.includes("="));
    const cookieStr = cookies.join("; ");

    const r2 = await fetch("https://query1.finance.yahoo.com/v1/test/getcrumb", {
      headers: { "User-Agent": UA, Cookie: cookieStr }, cache: "no-store",
    });
    const crumb = (await r2.text()).trim();
    if (!crumb || crumb.length > 60) return null;

    const r3 = await fetch(
      `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${code}.T?modules=assetProfile&crumb=${encodeURIComponent(crumb)}`,
      { headers: { "User-Agent": UA, Cookie: cookieStr }, cache: "no-store" }
    );
    if (!r3.ok) return null;
    const data = await r3.json();
    const desc = data?.quoteSummary?.result?.[0]?.assetProfile?.longBusinessSummary as string | undefined;
    return desc && desc.length > 20 ? desc : null;
  } catch {
    return null;
  }
}

export async function GET(
  _: Request,
  { params }: { params: Promise<{ code: string }> }
) {
  const { code } = await params;
  const url = new URL(_.url);
  const name   = url.searchParams.get("name")   ?? code;
  const sector = url.searchParams.get("sector") ?? "";

  // キャッシュ確認
  const cached = await getCached(code);
  if (cached) return NextResponse.json({ description: cached });

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return NextResponse.json({ description: null });

  const client = new Anthropic({ apiKey });

  // Yahoo Finance の英語説明文を取得（翻訳素材として使う）
  const enDesc = await getYahooDescription(code);

  const prompt = enDesc
    ? `以下は日本の上場企業「${name}」(銘柄コード: ${code}、セクター: ${sector || "不明"})の英語の事業説明文です。これを自然な日本語に翻訳してください。150字以内で簡潔にまとめてください。説明文だけ出力し、他の文章は不要です。\n\n${enDesc}`
    : `日本の上場企業「${name}」(銘柄コード: ${code}、セクター: ${sector || "不明"})がどのような事業を行っている会社か、100〜150字の日本語で説明してください。説明文だけ出力し、他の文章は不要です。`;

  try {
    const msg = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 256,
      messages: [{ role: "user", content: prompt }],
    });
    const description = (msg.content[0] as { type: string; text: string }).text?.trim() ?? null;
    if (description) await saveCache(code, description);
    return NextResponse.json({ description });
  } catch {
    return NextResponse.json({ description: null });
  }
}

import { NextRequest, NextResponse } from "next/server";

const SB_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SB_KEY = process.env.SUPABASE_SERVICE_KEY!;

function sbHeaders() {
  return {
    apikey: SB_KEY,
    Authorization: `Bearer ${SB_KEY}`,
    "Content-Type": "application/json",
    Prefer: "resolution=merge-duplicates",
  };
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { endpoint, keys } = body;

  if (!endpoint || !keys) {
    return NextResponse.json({ error: "invalid body" }, { status: 400 });
  }

  const res = await fetch(`${SB_URL}/rest/v1/app_push_subscriptions`, {
    method: "POST",
    headers: sbHeaders(),
    body: JSON.stringify({ endpoint, keys, enabled: true }),
  });

  if (!res.ok) {
    const text = await res.text();
    console.error("[push/subscribe]", text);
    return NextResponse.json({ error: text }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}

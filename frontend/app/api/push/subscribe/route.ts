import { NextRequest, NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { endpoint, keys } = body;

  if (!endpoint || !keys) {
    return NextResponse.json({ error: "invalid body" }, { status: 400 });
  }

  const sb = supabaseAdmin();
  const { error } = await sb.from("push_subscriptions").upsert(
    { endpoint, keys, enabled: true },
    { onConflict: "endpoint" }
  );

  if (error) {
    console.error("[push/subscribe]", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}

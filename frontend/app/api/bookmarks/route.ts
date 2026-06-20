import { NextRequest, NextResponse } from "next/server";
import { sbUrl, svcHeaders } from "@/lib/supabase";

// 匿名 client_id ごとのブックマークを Supabase(web_bookmarks) に永続化する。
// 認証は無いため client_id（ブラウザ発行の UUID）が唯一の識別子。
// 書き込みは service key が必要なため、すべて API ルート経由で行う。

function badRequest(msg: string) {
  return NextResponse.json({ error: msg }, { status: 400 });
}

// GET /api/bookmarks?clientId=...  → { codes: string[] }
export async function GET(req: NextRequest) {
  const clientId = req.nextUrl.searchParams.get("clientId")?.trim();
  if (!clientId) return badRequest("clientId required");

  const res = await fetch(
    sbUrl(`web_bookmarks?client_id=eq.${encodeURIComponent(clientId)}&select=code&order=created_at.asc`),
    { headers: svcHeaders(), cache: "no-store" },
  );
  if (!res.ok) {
    const text = await res.text();
    console.error("[bookmarks GET]", text);
    return NextResponse.json({ codes: [] }, { status: 500 });
  }
  const rows = (await res.json()) as { code: string }[];
  return NextResponse.json({ codes: rows.map((r) => r.code) });
}

// POST /api/bookmarks  { clientId, codes: string[] }  → upsert（複数可）
export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  const clientId = typeof body?.clientId === "string" ? body.clientId.trim() : "";
  const codes: string[] = Array.isArray(body?.codes)
    ? body.codes.filter((c: unknown): c is string => typeof c === "string")
    : [];
  if (!clientId) return badRequest("clientId required");
  if (codes.length === 0) return NextResponse.json({ ok: true });

  const rows = codes.map((code) => ({ client_id: clientId, code }));
  const res = await fetch(sbUrl("web_bookmarks"), {
    method: "POST",
    headers: svcHeaders({ Prefer: "resolution=merge-duplicates" }),
    body: JSON.stringify(rows),
  });
  if (!res.ok) {
    const text = await res.text();
    console.error("[bookmarks POST]", text);
    return NextResponse.json({ error: text }, { status: 500 });
  }
  return NextResponse.json({ ok: true });
}

// DELETE /api/bookmarks  { clientId, code }  → 1件削除
export async function DELETE(req: NextRequest) {
  const body = await req.json().catch(() => null);
  const clientId = typeof body?.clientId === "string" ? body.clientId.trim() : "";
  const code = typeof body?.code === "string" ? body.code.trim() : "";
  if (!clientId || !code) return badRequest("clientId and code required");

  const res = await fetch(
    sbUrl(
      `web_bookmarks?client_id=eq.${encodeURIComponent(clientId)}&code=eq.${encodeURIComponent(code)}`,
    ),
    { method: "DELETE", headers: svcHeaders() },
  );
  if (!res.ok) {
    const text = await res.text();
    console.error("[bookmarks DELETE]", text);
    return NextResponse.json({ error: text }, { status: 500 });
  }
  return NextResponse.json({ ok: true });
}

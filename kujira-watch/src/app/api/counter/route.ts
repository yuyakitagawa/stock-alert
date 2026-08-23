import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase";

// ページ表示1回につき1回だけ加算する（ヘッダーのVisitCounterが呼ぶ）。
export async function POST() {
  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase.rpc("increment_blog_visit_counter");

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ count: data as number });
}

// 加算せずに現在値だけ返す。ハンバーガーメニュー内の表示など、開くたびに
// POSTすると二重計上になる場所から使う。
export async function GET() {
  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase
    .from("blog_visit_counter")
    .select("count")
    .eq("id", 1)
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ count: data.count as number });
}

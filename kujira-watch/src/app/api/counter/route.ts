import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase";

export async function POST() {
  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase.rpc("increment_blog_visit_counter");

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ count: data as number });
}

import { createClient } from "@supabase/supabase-js";

// サーバー専用（Route Handler内でのみ使用）。SERVICE_KEYはクライアントに送られない。
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY;

export function getSupabaseServerClient() {
  if (!supabaseUrl || !supabaseServiceKey) {
    throw new Error("SUPABASE_URL と SUPABASE_SERVICE_KEY を設定してください");
  }
  return createClient(supabaseUrl, supabaseServiceKey, {
    auth: { persistSession: false },
  });
}

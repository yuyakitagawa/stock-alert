// fetch-based Supabase REST helpers (no JS client)
// .trim() で secret 末尾の改行/空白を除去（CI secret が改行付きだと
// URL が "https://...co\n/rest/..." となり Invalid URL で落ちるため）
const SB_URL  = (process.env.NEXT_PUBLIC_SUPABASE_URL ?? "").trim();
const SB_ANON = (process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "").trim();
const SB_SVC  = (process.env.SUPABASE_SERVICE_KEY ?? "").trim();

export function anonHeaders(): Record<string, string> {
  return {
    apikey:        SB_ANON,
    Authorization: `Bearer ${SB_ANON}`,
  };
}

export function svcHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return {
    apikey:           SB_SVC,
    Authorization:    `Bearer ${SB_SVC}`,
    "Content-Type":   "application/json",
    ...extra,
  };
}

export function sbUrl(path: string): string {
  return `${SB_URL}/rest/v1/${path}`;
}

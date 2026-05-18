// fetch-based Supabase REST helpers (no JS client)
const SB_URL  = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SB_ANON = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const SB_SVC  = process.env.SUPABASE_SERVICE_KEY!;

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

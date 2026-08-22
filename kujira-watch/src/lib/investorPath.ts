// 投資家ページのURL。クライアントコンポーネント（検索ボックス・ランキング）からも使うため、
// Supabaseクライアントを含む lib/investors.ts とは分けてある。
export function investorPath(filerId: number | null | undefined, filerName: string): string {
  // 採番前（理論上は起こらない。INSERTトリガーで採番される）は旧形式に落とし、
  // ページ側の名前→ID解決に任せる。
  return filerId != null ? `/investors/${filerId}` : `/investors/${encodeURIComponent(filerName)}`;
}

// EDINETの受付は17:15で終了し（全期間の実データで18時以降の提出はゼロ）、
// edinet_blog.ymlの18時・19時の便で当日ぶんが出揃う。この時刻を過ぎた日は
// 「もうこれ以上増えない＝確定」と言い切れる（デイトレ層向けの鮮度表示に使う）。
const DISCLOSURES_FIXED_HOUR_JST = 18;

// サーバーのタイムゾーンに関わらずJSTで判定する（Vercelの実行環境はUTC）。
function jstNow(): { date: string; hour: number } {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    hour12: false,
  }).formatToParts(new Date());
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  return { date: `${get("year")}-${get("month")}-${get("day")}`, hour: Number(get("hour")) };
}

// 指定日の開示が出揃っているか。過去日なら常に確定、当日は受付終了後に確定。
export function areDisclosuresFixed(discDate: string): boolean {
  const { date: today, hour } = jstNow();
  return discDate.slice(0, 10) < today || hour >= DISCLOSURES_FIXED_HOUR_JST;
}

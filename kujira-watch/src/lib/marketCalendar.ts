// 日本の「国民の祝日」を法令のルールから算出し、行政機関の休日を除いた営業日を数える。
//
// 大量保有報告書・変更報告書の提出期限（金融商品取引法27条の23第1項・27条の25第1項）は
// 「五日（日曜日その他政令で定める休日の日数は算入しない）以内」と定められている。
// この「算入しない日」は行政機関の休日に関する法律1条1項が定める
// 土曜日・日曜日・国民の祝日・12月29日〜1月3日と一致するため、ここではその定義で営業日を数える。
//
// 祝日を年ごとの表で持たないのは、表は必ず古くなるため。国民の祝日に関する法律の
// ルール（固定日・ハッピーマンデー・春分秋分・振替休日・国民の休日）をそのまま実装して、
// 何年先の日付でも同じコードで答えが出るようにしている。
//
// 対応年: 2023年以降。天皇誕生日が2月23日になったのは2020年から、スポーツの日が
// 10月第2月曜に戻ったのは2022年からで、それ以前と2020〜2021年の五輪特例（祝日の移動）は
// 実装していない。過去の日付を入力されたときは呼び出し側で弾く。
export const CALENDAR_MIN_YEAR = 2023;

/** Date を "YYYY-MM-DD"（日本時間の暦日として扱う）に変換する。 */
export function toIsoDate(date: Date): string {
  const y = date.getUTCFullYear();
  const m = String(date.getUTCMonth() + 1).padStart(2, "0");
  const d = String(date.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/** "YYYY-MM-DD" を UTC正午の Date にする。時差でも日付がずれないよう常にUTCで持つ。 */
export function parseIsoDate(iso: string): Date | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return null;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const date = new Date(Date.UTC(y, mo - 1, d));
  // 2026-02-30 のような存在しない日付は Date が繰り上げてしまうので、往復させて弾く。
  return toIsoDate(date) === iso ? date : null;
}

function addDays(date: Date, days: number): Date {
  return new Date(date.getTime() + days * 86400000);
}

/** その月のn番目の指定曜日（第2月曜など）。weekday は 0=日曜。 */
function nthWeekday(year: number, month: number, weekday: number, nth: number): string {
  const first = new Date(Date.UTC(year, month - 1, 1));
  const shift = (weekday - first.getUTCDay() + 7) % 7;
  return toIsoDate(new Date(Date.UTC(year, month - 1, 1 + shift + (nth - 1) * 7)));
}

// 春分日・秋分日の近似式（1980〜2099年で実際の官報と一致する、天文年鑑等で使われる式）。
// 祝日は前年2月の官報で正式に決まるため、遠い将来の日付は「見込み」であることを
// 呼び出し側の画面で断っている。
function equinoxDay(year: number, base: number): number {
  return Math.floor(base + 0.242194 * (year - 1980) - Math.floor((year - 1980) / 4));
}

function pad(month: number, day: number): string {
  return `${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

/**
 * その年の国民の祝日（振替休日・国民の休日を含む）を "YYYY-MM-DD" => 名称 で返す。
 */
export function japaneseHolidays(year: number): Map<string, string> {
  const base = new Map<string, string>();
  const put = (monthDay: string, name: string) => base.set(`${year}-${monthDay}`, name);

  put(pad(1, 1), "元日");
  base.set(nthWeekday(year, 1, 1, 2), "成人の日");
  put(pad(2, 11), "建国記念の日");
  put(pad(2, 23), "天皇誕生日");
  put(pad(3, equinoxDay(year, 20.8431)), "春分の日");
  put(pad(4, 29), "昭和の日");
  put(pad(5, 3), "憲法記念日");
  put(pad(5, 4), "みどりの日");
  put(pad(5, 5), "こどもの日");
  base.set(nthWeekday(year, 7, 1, 3), "海の日");
  put(pad(8, 11), "山の日");
  base.set(nthWeekday(year, 9, 1, 3), "敬老の日");
  put(pad(9, equinoxDay(year, 23.2488)), "秋分の日");
  base.set(nthWeekday(year, 10, 1, 2), "スポーツの日");
  put(pad(11, 3), "文化の日");
  put(pad(11, 23), "勤労感謝の日");

  const holidays = new Map(base);

  // 振替休日（国民の祝日に関する法律3条2項）。祝日が日曜のとき、その日後の最も近い
  // 祝日でない日を休日にする。5月3〜5日のように祝日が続く年は2日以上ずれる。
  for (const [iso] of [...base].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    const date = parseIsoDate(iso);
    if (!date || date.getUTCDay() !== 0) continue;
    let next = addDays(date, 1);
    while (base.has(toIsoDate(next))) next = addDays(next, 1);
    holidays.set(toIsoDate(next), "振替休日");
  }

  // 国民の休日（同3条3項）。前日と翌日がともに祝日の平日を休日にする。
  // 実際に発生するのは敬老の日と秋分の日が1日空くシルバーウィークの年。
  for (const [iso] of base) {
    const date = parseIsoDate(iso);
    if (!date) continue;
    const gap = addDays(date, 1);
    const gapIso = toIsoDate(gap);
    if (base.has(gapIso) || gap.getUTCDay() === 0) continue;
    if (base.has(toIsoDate(addDays(date, 2)))) holidays.set(gapIso, "国民の休日");
  }

  return holidays;
}

/** 年末年始の閉庁日（行政機関の休日に関する法律1条1項3号: 12月29日〜1月3日）。 */
function isYearEndClosure(date: Date): boolean {
  const month = date.getUTCMonth() + 1;
  const day = date.getUTCDate();
  return (month === 12 && day >= 29) || (month === 1 && day <= 3);
}

/** 提出期限の計算で「算入しない日」＝土日・祝日・年末年始か。 */
export function isNonBusinessDay(date: Date): boolean {
  const weekday = date.getUTCDay();
  if (weekday === 0 || weekday === 6) return true;
  if (isYearEndClosure(date)) return true;
  return japaneseHolidays(date.getUTCFullYear()).has(toIsoDate(date));
}

export type BusinessDayStep = { date: string; label: string };

/**
 * 起算日の翌日から数えてn営業日後の日付を返す（初日不算入・民法140条）。
 * steps には数えた営業日を順に入れて返し、画面で「何日をどう数えたか」を出せるようにする。
 */
export function addBusinessDays(start: Date, days: number): { deadline: Date; steps: BusinessDayStep[] } {
  const steps: BusinessDayStep[] = [];
  let cursor = start;
  let counted = 0;
  // 年末年始と大型連休が重なっても届く上限。無限ループの保険。
  for (let guard = 0; counted < days && guard < 400; guard += 1) {
    cursor = addDays(cursor, 1);
    if (isNonBusinessDay(cursor)) continue;
    counted += 1;
    steps.push({ date: toIsoDate(cursor), label: `${counted}営業日目` });
  }
  return { deadline: cursor, steps };
}

/** その日が休みなら理由を返す（画面の説明用）。営業日なら null。 */
export function nonBusinessReason(date: Date): string | null {
  const weekday = date.getUTCDay();
  if (weekday === 0) return "日曜日";
  if (weekday === 6) return "土曜日";
  if (isYearEndClosure(date)) return "年末年始（行政機関の休日）";
  return japaneseHolidays(date.getUTCFullYear()).get(toIsoDate(date)) ?? null;
}

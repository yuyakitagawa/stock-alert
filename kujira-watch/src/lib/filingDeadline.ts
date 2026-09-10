import { addBusinessDays, type BusinessDayStep, parseIsoDate, toIsoDate } from "@/lib/marketCalendar";

// 大量保有報告書・変更報告書の「出す義務があるか」「いつまでに出すか」を判定する。
// /tools/filing-deadline の計算本体。
//
// 根拠条文:
//   - 新規（大量保有報告書）: 金融商品取引法27条の23第1項。株券等保有割合が5%を超えた者は
//     「五日（日曜日その他政令で定める休日の日数は算入しない）以内」に提出する。
//   - 変更報告書: 同法27条の25第1項。保有割合が1%以上増加または減少した場合等に、同じく5日以内。
//   - 特例報告: 同法27条の26。金融商品取引業者・銀行・保険会社等が、重要提案行為等を
//     目的とせず保有割合10%以下で保有する場合、基準日（同条3項・施行令14条の7の2）から
//     5日以内にまとめて提出できる。基準日は毎月2回以上、政令で定める日。
//
// 「五日」の起算は初日不算入（民法140条）で、義務が生じた日の翌日から数える。
// 算入しない休日は行政機関の休日（土日・国民の祝日・12/29〜1/3）とした。lib/marketCalendar.ts 参照。

/** 大量保有報告制度のしきい値（%）。 */
export const THRESHOLD_PCT = 5;
/** 変更報告書が必要になる保有割合の増減幅（ポイント）。 */
export const CHANGE_STEP_PT = 1;
/** 提出までの日数（休日を算入しない5日）。 */
export const FILING_DAYS = 5;

export type FilingJudgement = {
  /** 提出義務の種類。 */
  kind: "new" | "change" | "exit" | "none";
  headline: string;
  detail: string;
};

/**
 * 前回報告時と今回の保有割合から、提出義務の有無と種類を判定する。
 * prevRatioPct が null なら「まだ一度も報告していない」＝新規の判定だけを行う。
 */
export function judgeFiling(input: {
  prevRatioPct: number | null;
  ratioPct: number;
}): FilingJudgement {
  const { prevRatioPct, ratioPct } = input;

  if (prevRatioPct === null) {
    if (ratioPct > THRESHOLD_PCT) {
      return {
        kind: "new",
        headline: "大量保有報告書の提出義務あり",
        detail: `保有割合が${THRESHOLD_PCT}%を超えたため、新規の大量保有報告書（法27条の23第1項）の提出対象です。`,
      };
    }
    return {
      kind: "none",
      headline: "提出義務なし",
      detail: `保有割合が${THRESHOLD_PCT}%以下のため、大量保有報告書の提出対象ではありません。${THRESHOLD_PCT}%ちょうどは「超えた」に当たらず対象外です。`,
    };
  }

  const diff = ratioPct - prevRatioPct;

  // 前回が5%以下＝まだ報告義務者になっていない状態からの判定は新規と同じ。
  if (prevRatioPct <= THRESHOLD_PCT) {
    return judgeFiling({ prevRatioPct: null, ratioPct });
  }

  if (ratioPct <= THRESHOLD_PCT) {
    return {
      kind: "exit",
      headline: "変更報告書（5%以下になった旨）の提出義務あり",
      detail: `保有割合が${prevRatioPct}%から${ratioPct}%へ下がり、${THRESHOLD_PCT}%以下になりました。この報告をもって以後の提出義務は終わります。`,
    };
  }

  if (Math.abs(diff) >= CHANGE_STEP_PT) {
    const direction = diff > 0 ? "増加" : "減少";
    return {
      kind: "change",
      headline: "変更報告書の提出義務あり",
      detail: `保有割合が${Math.abs(diff).toFixed(2)}ポイント${direction}し、${CHANGE_STEP_PT}%以上の変動（法27条の25第1項）に当たります。`,
    };
  }

  return {
    kind: "none",
    headline: "提出義務なし",
    detail: `保有割合の変動が${Math.abs(diff).toFixed(2)}ポイントで、${CHANGE_STEP_PT}%に届きません。ただし保有目的や重要な契約が変われば、変動が${CHANGE_STEP_PT}%未満でも変更報告書が必要になります。`,
  };
}

export type DeadlineResult = {
  baseDate: string;
  deadline: string;
  steps: BusinessDayStep[];
};

/** 義務が生じた日（基準日）から、休日を算入しない5日後を返す。 */
export function filingDeadline(baseIso: string, days = FILING_DAYS): DeadlineResult | null {
  const base = parseIsoDate(baseIso);
  if (!base) return null;
  const { deadline, steps } = addBusinessDays(base, days);
  return { baseDate: baseIso, deadline: toIsoDate(deadline), steps };
}

/**
 * 特例報告の基準日（施行令14条の7の2: 毎月15日と月末）のうち、指定日以後で最も近い日を返す。
 * 特例報告は「取引のたび」ではなく基準日ごとにまとめて報告する。
 */
export function nextSpecialReportBaseDate(iso: string): string | null {
  const date = parseIsoDate(iso);
  if (!date) return null;
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth();
  const day = date.getUTCDate();
  if (day <= 15) return toIsoDate(new Date(Date.UTC(year, month, 15)));
  // 月末は翌月0日で求める（うるう年・月の大小を自前で持たない）。
  return toIsoDate(new Date(Date.UTC(year, month + 1, 0)));
}

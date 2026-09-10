"use client";

import { useMemo, useState } from "react";
import { filingDeadline, judgeFiling, nextSpecialReportBaseDate, FILING_DAYS } from "@/lib/filingDeadline";
import { CALENDAR_MIN_YEAR, nonBusinessReason, parseIsoDate } from "@/lib/marketCalendar";

// 大量保有報告書・変更報告書の提出義務と提出期限を出す。計算はすべてブラウザ内で完結し、
// 入力値はどこにも送らない（保有割合は公開前の情報になりうるため、送信しない設計にしている）。

const KIND_TONE: Record<string, string> = {
  new: "text-loss",
  change: "text-loss",
  exit: "text-ink-secondary",
  none: "text-ink-secondary",
};

function formatJaDate(iso: string): string {
  const date = parseIsoDate(iso);
  if (!date) return iso;
  const weekday = ["日", "月", "火", "水", "木", "金", "土"][date.getUTCDay()];
  return `${date.getUTCFullYear()}年${date.getUTCMonth() + 1}月${date.getUTCDate()}日（${weekday}）`;
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-bold text-ink-secondary">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-2xs text-ink-tertiary">{hint}</span>}
    </label>
  );
}

const inputClass =
  "w-full rounded-md border border-rule-strong bg-paper px-3 py-2 text-base text-ink tabular-nums outline-none focus:border-brand-blue";

export default function FilingDeadlineCalculator() {
  const [baseDate, setBaseDate] = useState("");
  const [ratio, setRatio] = useState("");
  const [prevRatio, setPrevRatio] = useState("");

  const result = useMemo(() => {
    const ratioValue = Number.parseFloat(ratio);
    if (!Number.isFinite(ratioValue) || ratioValue < 0 || ratioValue > 100) return null;
    const prevValue = prevRatio.trim() === "" ? null : Number.parseFloat(prevRatio);
    if (prevValue !== null && (!Number.isFinite(prevValue) || prevValue < 0 || prevValue > 100)) return null;

    const judgement = judgeFiling({ prevRatioPct: prevValue, ratioPct: ratioValue });
    const date = parseIsoDate(baseDate);
    if (!date || date.getUTCFullYear() < CALENDAR_MIN_YEAR) {
      return { judgement, deadline: null, baseReason: null, specialBase: null };
    }
    return {
      judgement,
      deadline: judgement.kind === "none" ? null : filingDeadline(baseDate),
      baseReason: nonBusinessReason(date),
      specialBase: nextSpecialReportBaseDate(baseDate),
    };
  }, [baseDate, ratio, prevRatio]);

  return (
    <section className="mb-8 rounded-md border border-rule bg-section-tint p-4 sm:p-5">
      <h2 className="mb-4 text-lg font-bold text-brand-navy">入力</h2>
      <div className="grid gap-4 sm:grid-cols-3">
        <Field label="義務が生じた日" hint={`${CALENDAR_MIN_YEAR}年以降の日付`}>
          <input
            type="date"
            className={inputClass}
            value={baseDate}
            min={`${CALENDAR_MIN_YEAR}-01-01`}
            onChange={(e) => setBaseDate(e.target.value)}
          />
        </Field>
        <Field label="今回の株券等保有割合（%）" hint="潜在株式を含めた割合">
          <input
            type="number"
            inputMode="decimal"
            step="0.01"
            min="0"
            max="100"
            placeholder="5.12"
            className={inputClass}
            value={ratio}
            onChange={(e) => setRatio(e.target.value)}
          />
        </Field>
        <Field label="前回報告時の保有割合（%）" hint="初めての報告なら空欄">
          <input
            type="number"
            inputMode="decimal"
            step="0.01"
            min="0"
            max="100"
            placeholder="空欄＝新規"
            className={inputClass}
            value={prevRatio}
            onChange={(e) => setPrevRatio(e.target.value)}
          />
        </Field>
      </div>

      {!result && (
        <p className="mb-0 mt-5 text-sm text-ink-tertiary">
          保有割合を入れると判定が出ます。日付も入れると提出期限まで計算します。
        </p>
      )}

      {result && (
        <div className="mt-5 border-t border-rule pt-5">
          <p className={`mb-1 text-lg font-bold ${KIND_TONE[result.judgement.kind]}`}>
            {result.judgement.headline}
          </p>
          <p className="mb-4 text-sm leading-relaxed text-ink-secondary">{result.judgement.detail}</p>

          {result.deadline && (
            <>
              <dl className="m-0 grid grid-cols-2 gap-x-4 gap-y-3">
                <div>
                  <dt className="text-2xs text-ink-muted">起算日</dt>
                  <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-ink">
                    {formatJaDate(result.deadline.baseDate)}
                    {result.baseReason && (
                      <span className="ml-1 text-xs font-normal text-ink-tertiary">
                        （{result.baseReason}）
                      </span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-2xs text-ink-muted">提出期限</dt>
                  <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-loss">
                    {formatJaDate(result.deadline.deadline)}
                  </dd>
                </div>
              </dl>

              <details className="mt-4 rounded-md border border-rule bg-paper p-3">
                <summary className="cursor-pointer text-xs font-bold text-ink-secondary">
                  {FILING_DAYS}日をどう数えたか
                </summary>
                <ol className="mb-0 mt-2 list-decimal pl-5 text-xs text-ink-secondary">
                  {result.deadline.steps.map((step) => (
                    <li key={step.date} className="tabular-nums">
                      {formatJaDate(step.date)}
                    </li>
                  ))}
                </ol>
                <p className="mb-0 mt-2 text-2xs leading-relaxed text-ink-tertiary">
                  起算日の翌日から数え、土曜・日曜・国民の祝日・12月29日〜1月3日は算入していません。
                </p>
              </details>
            </>
          )}

          {result.specialBase && result.judgement.kind !== "none" && (
            <p className="mb-0 mt-4 text-xs leading-relaxed text-ink-tertiary">
              特例報告（法27条の26）の対象になる金融機関・運用会社の場合は、取引日ではなく基準日ごとの報告になります。
              この日付の直後の基準日は{formatJaDate(result.specialBase)}で、そこから同じく{FILING_DAYS}日以内が期限です。
            </p>
          )}
        </div>
      )}
    </section>
  );
}

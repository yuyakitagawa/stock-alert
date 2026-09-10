"use client";

import { useMemo, useState } from "react";
import { buybackImpactVerdict, calcBuybackImpact } from "@/lib/buybackImpact";

// 自社株買いの取得枠から需給・EPSへの影響を出す。計算はブラウザ内で完結する。
// 入力単位は決定開示の書き方に合わせて億円・万株にしてある（円・株で打つと桁を間違える）。

const OKU = 100_000_000;
const MAN = 10_000;

function Field({
  label,
  unit,
  hint,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  unit: string;
  hint?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-bold text-ink-secondary">
        {label}
        <span className="ml-1 font-normal text-ink-tertiary">（{unit}）</span>
      </span>
      <input
        type="number"
        inputMode="decimal"
        step="any"
        min="0"
        placeholder={placeholder}
        className="w-full rounded-md border border-rule-strong bg-paper px-3 py-2 text-base text-ink tabular-nums outline-none focus:border-brand-blue"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      {hint && <span className="mt-1 block text-2xs text-ink-tertiary">{hint}</span>}
    </label>
  );
}

function num(value: string): number {
  return Number.parseFloat(value);
}

function fmt(value: number, digits = 2): string {
  return value.toLocaleString("ja-JP", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function BuybackImpactCalculator() {
  const [maxAmount, setMaxAmount] = useState("");
  const [maxShares, setMaxShares] = useState("");
  const [outstanding, setOutstanding] = useState("");
  const [price, setPrice] = useState("");
  const [netIncome, setNetIncome] = useState("");
  const [volume, setVolume] = useState("");

  const result = useMemo(
    () =>
      calcBuybackImpact({
        maxAmountYen: num(maxAmount) * OKU,
        maxShares: num(maxShares) * MAN,
        sharesOutstanding: num(outstanding) * MAN,
        price: num(price),
        netIncomeYen: netIncome.trim() === "" ? undefined : num(netIncome) * OKU,
        avgDailyVolume: volume.trim() === "" ? undefined : num(volume) * MAN,
      }),
    [maxAmount, maxShares, outstanding, price, netIncome, volume]
  );

  return (
    <section className="mb-8 rounded-md border border-rule bg-section-tint p-4 sm:p-5">
      <h2 className="mb-4 text-lg font-bold text-brand-navy">入力</h2>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="取得価額の総額の上限" unit="億円" value={maxAmount} onChange={setMaxAmount} placeholder="50" />
        <Field label="取得する株式の総数の上限" unit="万株" value={maxShares} onChange={setMaxShares} placeholder="300" />
        <Field
          label="発行済株式総数"
          unit="万株"
          hint="自己株式を除いた数（決定開示に併記されています）"
          value={outstanding}
          onChange={setOutstanding}
          placeholder="5000"
        />
        <Field label="株価" unit="円" value={price} onChange={setPrice} placeholder="1800" />
        <Field label="当期純利益（任意）" unit="億円" value={netIncome} onChange={setNetIncome} placeholder="120" />
        <Field label="1日平均出来高（任意）" unit="万株" value={volume} onChange={setVolume} placeholder="20" />
      </div>

      {!result && (
        <p className="mb-0 mt-5 text-sm text-ink-tertiary">
          上の4つ（上限金額・上限株数・発行済株式総数・株価）を入れると計算されます。
        </p>
      )}

      {result && (
        <div className="mt-5 border-t border-rule pt-5">
          <p className="mb-4 text-lg font-bold text-brand-navy">{buybackImpactVerdict(result)}</p>
          <dl className="m-0 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            <div>
              <dt className="text-2xs text-ink-muted">発行済株式に対する比率</dt>
              <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-ink">{fmt(result.sharesRatioPct)}%</dd>
            </div>
            <div>
              <dt className="text-2xs text-ink-muted">実際に取得しうる株数</dt>
              <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-ink">
                {fmt(result.effectiveShares / MAN, 1)}万株
                <span className="ml-1 text-xs font-normal text-ink-tertiary">
                  {result.bindingLimit === "amount" ? "金額の上限が先" : "株数の上限が先"}
                </span>
              </dd>
            </div>
            <div>
              <dt className="text-2xs text-ink-muted">取得枠 ÷ 時価総額</dt>
              <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-ink">{fmt(result.marketCapRatioPct)}%</dd>
            </div>
            <div>
              <dt className="text-2xs text-ink-muted">EPS押し上げ率</dt>
              <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-gain">+{fmt(result.epsUpliftPct)}%</dd>
            </div>
            {result.epsBefore !== null && result.epsAfter !== null && (
              <div className="col-span-2">
                <dt className="text-2xs text-ink-muted">1株当たり利益（取得前 → 取得後）</dt>
                <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-ink">
                  {fmt(result.epsBefore, 1)}円 → {fmt(result.epsAfter, 1)}円
                </dd>
              </div>
            )}
            {result.volumeDays !== null && (
              <div className="col-span-2">
                <dt className="text-2xs text-ink-muted">1日平均出来高の何日ぶんか</dt>
                <dd className="m-0 mt-0.5 text-sm font-bold tabular-nums text-ink">{fmt(result.volumeDays, 1)}日</dd>
              </div>
            )}
          </dl>
          <p className="mb-0 mt-4 text-2xs leading-relaxed text-ink-tertiary">
            取締役会が決議するのは上限であり、枠を使い切らないまま取得期間が終わることもあります。
            ここで出る数字は「枠を使い切った場合」の最大値です。EPS押し上げ率は利益が変わらない前提で、
            自己株式が1株当たり利益の分母から除かれることによる計算上の効果です。
          </p>
        </div>
      )}
    </section>
  );
}

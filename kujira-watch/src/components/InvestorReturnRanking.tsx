"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import DealTypeIcon from "./DealTypeIcon";
import MagnitudeBar from "./MagnitudeBar";
import DealTypeLabel from "./DealTypeLabel";
import { displayFilerName, formatDate } from "@/lib/format";
import {
  formatSignedPercent,
  formatSignedPoint,
  type InvestorReturnRow,
} from "@/lib/investorReturns";
import { DEAL_TYPES, type DealType } from "@/types/article";
import { investorPath } from "@/lib/investorPath";

// /ranking/returns のランキング本体＋投資家分類の絞り込み。
// 全件（2026-08-22時点で198名）を最初からクライアントに渡してその場で絞り込む。
// `?category=`のsearchParamsを読むとページがリクエストごとの動的レンダリングになり
// ISRキャッシュが効かなくなるため（/trendingの売買方向フィルターと同じ判断）。
// 分類ごとに「その分類の中での順位」を1位から振り直す。全体順位のまま歯抜けの数字を出しても
// 「アクティビストの中で何番目か」が読み取れないため。
// ボタンの見た目は一覧ページの絞り込みナビ（FilterButtonNav）に合わせているが、
// あちらはURL遷移のリンク、こちらはその場で切り替えるボタンなので別実装にしている。

const ALL = "すべて";
type Selection = typeof ALL | DealType;

export default function InvestorReturnRanking({
  rows,
  size,
  publishedFilerNames,
  publishedStockCodes,
}: {
  rows: InvestorReturnRow[];
  size: number;
  /** 公開している投資家ページの提出者名。ここに無い投資家はリンクにしない（404を作らない）。 */
  publishedFilerNames: string[];
  /** 公開している銘柄ページのコード。 */
  publishedStockCodes: string[];
}) {
  const [selected, setSelected] = useState<Selection>(ALL);
  // Setはサーバー→クライアントの境界を越えられないため配列で受け取ってここで組み立てる。
  const publishedFilers = useMemo(() => new Set(publishedFilerNames), [publishedFilerNames]);
  const publishedCodes = useMemo(() => new Set(publishedStockCodes), [publishedStockCodes]);

  // 該当0名の分類はボタンごと出さない（押しても空になるボタンを並べない）。
  const options = useMemo(() => {
    const counts = new Map<DealType, number>();
    for (const row of rows) counts.set(row.category, (counts.get(row.category) ?? 0) + 1);
    return [
      { value: ALL as Selection, label: `${ALL}（${rows.length}名）` },
      ...DEAL_TYPES.filter((type) => counts.has(type)).map((type) => ({
        value: type as Selection,
        label: `${type}（${counts.get(type)}名）`,
      })),
    ];
  }, [rows]);

  const filtered = useMemo(
    () => (selected === ALL ? rows : rows.filter((row) => row.category === selected)),
    [rows, selected]
  );
  const shown = filtered.slice(0, size);
  const filtering = selected !== ALL;
  // 量感バーの基準。マイナスの投資家も同じ長さの物差しで比べられるよう絶対値の最大を取る。
  const maxReturn = useMemo(
    () => shown.reduce((max, row) => Math.max(max, Math.abs(row.avgReturn)), 0),
    [shown]
  );

  return (
    <>
      <details className="mb-4" open={filtering}>
        <summary className="cursor-pointer text-sm text-brand-blue hover:underline">
          投資家の種類で絞り込む
          {filtering && `（現在: ${selected}）`}
        </summary>
        <Stack
          role="group"
          aria-label="投資家の種類で絞り込む"
          direction="row"
          className="no-scrollbar"
          sx={{
            gap: 1,
            mt: 1.5,
            // スマホでは折り返すと絞り込みだけで画面が埋まってしまうため横スクロールにする。
            flexWrap: { xs: "nowrap", sm: "wrap" },
            overflowX: { xs: "auto", sm: "visible" },
            "& .MuiButton-root": { flexShrink: 0 },
          }}
        >
          {options.map((option) => (
            <Button
              key={option.value}
              type="button"
              variant={option.value === selected ? "contained" : "outlined"}
              aria-pressed={option.value === selected}
              onClick={() => setSelected(option.value)}
            >
              {option.label}
            </Button>
          ))}
        </Stack>
      </details>
      <p className="kicker mb-2 text-ink-muted">
        順位・投資家 ／ 分類・買い開示件数・勝率・日経平均比・最も上がった銘柄・3ヶ月平均リターン
      </p>
      {/* 「1行目=順位＋投資家名（全幅）／2行目=メタ・平均リターン（右端）」の1件1カード。
          投資家・最高銘柄と別々のリンクが2つあるためカード全体はリンクにしない。 */}
      <ul className="card-grid card-grid-wide">
        {shown.map((row, index) => (
          <li key={row.filerName} className="card">
            <span className="flex items-start gap-2">
              <span className="w-5 shrink-0 font-bold tabular-nums text-ink-muted">
                {index + 1}
              </span>
              {/* 投資家のロゴは持てないため、分類の絵文字＋分類色のアイコンで代替する。 */}
              <DealTypeIcon dealType={row.category} />
              {publishedFilers.has(row.filerName) ? (
                <Link
                  href={investorPath(row.filerId, row.filerName)}
                  className="min-w-0 grow font-medium text-brand-blue [overflow-wrap:anywhere] hover:underline"
                >
                  {displayFilerName(row.filerName)}
                </Link>
              ) : (
                <span className="min-w-0 grow font-medium [overflow-wrap:anywhere]">
                  {displayFilerName(row.filerName)}
                </span>
              )}
            </span>
            <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-tertiary">
              {/* 分類で絞り込んでいる間は全行が同じ分類なのでラベルを繰り返さない。 */}
              {!filtering && <DealTypeLabel dealType={row.category} />}
              <span className="font-semibold">買い開示{row.positionCount}件</span>
              <span>勝率{row.winRate}%</span>
              {row.avgExcessReturn !== null && (
                <span>日経平均比{formatSignedPoint(row.avgExcessReturn)}</span>
              )}
              <span className="flex items-center gap-1">
                最高
                {publishedCodes.has(row.bestStockCode) ? (
                  <Link href={`/stocks/${row.bestStockCode}`} className="text-brand-blue hover:underline">
                    {row.bestStockName}（{formatSignedPercent(row.bestReturn, 0)}）
                  </Link>
                ) : (
                  <span>
                    {row.bestStockName}（{formatSignedPercent(row.bestReturn, 0)}）
                  </span>
                )}
              </span>
              <span>最終買い{formatDate(row.latestBuyDate)}</span>
              <span
                className={`ml-auto whitespace-nowrap text-sm font-semibold tabular-nums ${
                  row.avgReturn >= 0 ? "text-gain" : "text-loss"
                }`}
              >
                {formatSignedPercent(row.avgReturn)}
                <span className="kicker ml-1 font-normal text-ink-muted">3ヶ月平均</span>
              </span>
            </span>
            {/* 平均リターンの量感バー。1位だけ金色にして先頭が読み取れるようにする。 */}
            <MagnitudeBar
              value={Math.abs(row.avgReturn)}
              max={maxReturn}
              tone={index === 0 && row.avgReturn >= 0 ? "gold" : row.avgReturn >= 0 ? "gain" : "loss"}
            />
          </li>
        ))}
      </ul>
      <p className="mt-4 text-xs leading-relaxed text-ink-muted">
        {filtering
          ? `「${selected}」に分類される${filtered.length}名`
          : `集計対象の${rows.length}名`}
        のうち上位{shown.length}名を表示しています。
        {filtered.length > shown.length && `（${filtered.length - shown.length}名は表示していません）`}
      </p>
    </>
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import Box from "@mui/material/Box";
import DealTypeBadge from "@/components/DealTypeBadge";
import FilterButtonNav from "@/components/FilterButtonNav";
import { getFilerWinRates } from "@/lib/investors";
import { formatDate } from "@/lib/format";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { DEAL_TYPES, type DealType } from "@/types/article";
import AdUnit from "@/components/AdUnit";

export const revalidate = 3600;

// n(結果確定件数)がこれ未満の投資家は表示しない。n=1〜2の勝率は誤差が大きく、
// 「参考にできる」ランキングとして出すには信頼性が低すぎるため。
const MIN_N = 5;

const title = "投資家別 3ヶ月勝率ランキング";
const description =
  "EDINET大量保有報告書で買い増しを開示した投資家について、その回の推定取得金額をもとに" +
  "63営業日(3ヶ月)後までの株価騰落から損益を推定し、投資家別に合算したランキング。" +
  "過去の実績が良い投資家を参考にできます。";

function formatOku(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toLocaleString("ja-JP", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}億円`;
}

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/ranking` },
  openGraph: { title, description, url: `${SITE_URL}/ranking` },
};

type Props = {
  searchParams: Promise<{ category?: string }>;
};

export default async function RankingPage({ searchParams }: Props) {
  const { category } = await searchParams;
  const filers = await getFilerWinRates(MIN_N);

  const counts = new Map<DealType, number>();
  for (const filer of filers) {
    counts.set(filer.category, (counts.get(filer.category) ?? 0) + 1);
  }
  const activeCategories = DEAL_TYPES.filter((c) => (counts.get(c) ?? 0) > 0);
  const selectedCategory =
    category && DEAL_TYPES.includes(category as DealType) ? (category as DealType) : null;
  const visibleFilers = selectedCategory
    ? filers.filter((f) => f.category === selectedCategory)
    : filers;

  const updatedAt = filers[0]?.updatedAt;
  const holdDays = filers[0]?.holdDays ?? 63;

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: `${SITE_URL}/ranking` },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: title,
    itemListElement: visibleFilers.slice(0, 100).map((f, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: f.filerName,
      url: `${SITE_URL}/investors/${encodeURIComponent(f.filerName)}`,
    })),
  };

  return (
    <div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }}
      />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">{title}</span>
      </nav>
      <h1 className="mb-1 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
      {updatedAt && (
        <p className="kicker mb-3 text-brand-blue">最終更新: {formatDate(updatedAt)}</p>
      )}
      <p className="mb-4 text-sm text-foreground/50">
        投資家がEDINET大量保有報告書で保有比率を増やした（買い増し・新規取得）ことを開示するたびに、
        その回の推定取得金額（発行済株式数×株価×保有比率の変化幅）が{holdDays}営業日（約3ヶ月）後まで
        の株価騰落でいくら増減したかを推定し、投資家ごとに全開示分を合算したランキングです。
        過去の実績が良い投資家を参考にできます。投資助言ではありません。
      </p>
      <p className="mb-2 text-xs text-foreground/40">
        ※自己申告・訂正報告書・保有比率51%超・売却方向の開示、発行済株式数が取得できない開示は除外し、
        開示からまだ{holdDays}営業日経っていないものは結果未確定として集計から外しています。
      </p>
      <p className="mb-6 text-xs text-foreground/40">
        ※開示件数（n）が少ない投資家はブレが大きいため、{MIN_N}件未満の投資家は掲載していません。
      </p>
      {filers.length > 0 && (
        <FilterButtonNav
          ariaLabel="カテゴリで絞り込む"
          items={[
            {
              href: "/ranking",
              label: `すべて（${filers.length}件）`,
              selected: selectedCategory === null,
            },
            ...activeCategories.map((c) => ({
              href: `/ranking?category=${encodeURIComponent(c)}`,
              label: `${c}（${counts.get(c)}件）`,
              selected: selectedCategory === c,
            })),
          ]}
        />
      )}
      {visibleFilers.length === 0 ? (
        <p className="text-foreground/50">該当する投資家がいません。</p>
      ) : (
        // 5列テーブルは狭い画面で投資家名が1文字ずつ折り返されて読めないため、
        // 全画面幅で1件1行のリスト表示にする（分類は名前の下にバッジで添える）。
        <Box>
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              gap: 1,
              pb: 0.75,
              fontSize: "0.6875rem",
              letterSpacing: "0.08em",
              color: "text.disabled",
            }}
          >
            <span>順位・投資家</span>
            <span>トータルの推計リターン</span>
          </Box>
          {visibleFilers.map((f, index) => (
            <Box
              key={f.filerName}
              sx={{
                display: "flex",
                alignItems: "flex-start",
                gap: 1.25,
                py: 1.25,
                borderTop: 1,
                borderColor: "divider",
              }}
            >
              <Box
                sx={{
                  flexShrink: 0,
                  width: 22,
                  fontWeight: 700,
                  color: "text.disabled",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {index + 1}
              </Box>
              <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                <Link
                  href={`/investors/${encodeURIComponent(f.filerName)}`}
                  className="font-medium text-brand-blue hover:underline"
                  style={{ overflowWrap: "anywhere" }}
                >
                  {f.filerName}
                </Link>
                <Box sx={{ mt: 0.5, display: "flex", flexWrap: "wrap", alignItems: "center", gap: 1 }}>
                  <DealTypeBadge dealType={f.category} />
                  <Box component="span" sx={{ fontSize: "0.75rem", color: "text.secondary" }}>
                    {f.n}件
                  </Box>
                </Box>
              </Box>
              <Box
                sx={{
                  flexShrink: 0,
                  whiteSpace: "nowrap",
                  fontWeight: 600,
                  fontVariantNumeric: "tabular-nums",
                  color: f.totalReturnOku >= 0 ? "success.main" : "error.main",
                }}
              >
                {formatOku(f.totalReturnOku)}
              </Box>
            </Box>
          ))}
        </Box>
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}

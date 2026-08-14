import type { Metadata } from "next";
import Link from "next/link";
import FeaturedArticleCard from "@/components/FeaturedArticleCard";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { getRecentArticles } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { formatDate, formatDealAmount } from "@/lib/format";
import { buildWeeklySummary } from "@/lib/weeklyStats";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";

// 「大口投資家の動きを教えて」等の包括的な検索・LLMクエリに直答するための集約ページ。
// 直近7日間の開示を横断的に要約する（個別記事は取引ごとの解説に特化しているため、
// このページが唯一の「まとめて見る」導線になる）。
const WINDOW_DAYS = 7;

// 一覧を全件カード表示すると縦に長くなりすぎるため、金額規模が大きい上位3件のみ
// 「注目」として本文付きで見せ、残りは取引日ごとに/date/[date]へのリンクへ集約する。
const FEATURED_COUNT = 3;

export async function generateMetadata(): Promise<Metadata> {
  const { contents } = await getRecentArticles(WINDOW_DAYS);
  const title = "大口投資家の動きまとめ（直近7日間）";
  const description = `EDINET大量保有報告書をもとにした、直近${WINDOW_DAYS}日間の大口投資家（クジラ）の動きまとめ。${contents.length}件の開示を取引日順に一覧できます。`;
  const url = `${SITE_URL}/weekly`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: { title, description, url },
  };
}

export default async function WeeklyDigestPage() {
  const { contents } = await getRecentArticles(WINDOW_DAYS);
  const url = `${SITE_URL}/weekly`;

  const oldestDate = contents.length > 0 ? contents[contents.length - 1].dealDate : null;
  const newestDate = contents.length > 0 ? contents[0].dealDate : null;
  const summary = buildWeeklySummary(contents);
  const netAmount = summary.buyAmount - summary.sellAmount;
  const netLabel = netAmount >= 0 ? "買い越し" : "売り越し";
  const featured = [...contents].sort((a, b) => b.dealAmount - a.dealAmount).slice(0, FEATURED_COUNT);
  const dateGroups = groupArticlesByDealDate(contents);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "大口投資家の動きまとめ", item: url },
    ],
  };

  // 可視コンテンツと構造化データを一致させるため、本文で実際にリンクしている要素
  // （注目記事の個別ページ＋取引日別アーカイブページ）だけをItemList化する。
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "大口投資家の動きまとめ（直近7日間）",
    itemListElement: [
      ...featured.map((article, index) => ({
        "@type": "ListItem",
        position: index + 1,
        name: article.title,
        url: `${SITE_URL}/articles/${article.id}`,
      })),
      ...dateGroups.map((group, index) => ({
        "@type": "ListItem",
        position: featured.length + index + 1,
        name: `${group.label}の大口投資家の動き`,
        url: `${SITE_URL}/date/${group.date}`,
      })),
    ],
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
        <span className="text-foreground/70">大口投資家の動きまとめ</span>
      </nav>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">大口投資家の動き（直近7日間）</h1>
        {contents.length > 0 && oldestDate && newestDate ? (
          <p className="mt-3 text-sm leading-relaxed text-foreground/70">
            {SITE_NAME}がEDINET大量保有報告書をもとに集計した、{formatDate(oldestDate)}〜
            {formatDate(newestDate)}の大口投資家（クジラ）の動きです。この期間に{summary.totalCount}
            件の大量保有・変更報告書が開示され、推定取引金額の合計は約
            {formatDealAmount(summary.totalAmount)}でした（取得・売却双方向を合算した規模で、
            資金の純流入額ではありません）。
          </p>
        ) : (
          <p className="mt-3 text-sm leading-relaxed text-foreground/70">
            直近{WINDOW_DAYS}日間はEDINET大量保有報告書の新規開示がありませんでした。
          </p>
        )}
      </div>

      {contents.length > 0 && (
        <section className="mb-10 border-y border-rule py-6">
          <h2 className="text-lg font-bold text-brand-navy">今週のポイント</h2>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="bg-section-tint p-4">
              <p className="kicker text-foreground/50">買い</p>
              <p className="mt-1 text-xl font-bold text-brand-navy">{summary.buyCount}件</p>
              <p className="mt-0.5 text-sm text-foreground/60">{formatDealAmount(summary.buyAmount)}</p>
            </div>
            <div className="bg-section-tint p-4">
              <p className="kicker text-foreground/50">売り</p>
              <p className="mt-1 text-xl font-bold text-brand-navy">{summary.sellCount}件</p>
              <p className="mt-0.5 text-sm text-foreground/60">{formatDealAmount(summary.sellAmount)}</p>
            </div>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-foreground/70">
            金額ベースでは買いが{formatDealAmount(summary.buyAmount)}、売りが
            {formatDealAmount(summary.sellAmount)}で、差し引き{formatDealAmount(Math.abs(netAmount))}の
            {netLabel}でした（複数の開示を合算した推定値のため、実際の資金フローとは異なります）。
            {summary.topCategories.length > 0 && (
              <>
                取引額で最も存在感が大きかった投資家分類は
                {summary.topCategories.map((c, i) => (
                  <span key={c.dealType}>
                    {i > 0 && "、"}
                    <span title={DEAL_TYPE_DESCRIPTIONS[c.dealType]} className="font-bold text-brand-navy">
                      {c.dealType}
                    </span>
                    （{c.count}件・{formatDealAmount(c.amount)}）
                  </span>
                ))}
                でした。
              </>
            )}
            {summary.topStocks.length > 0 && (
              <>
                個別銘柄では
                {summary.topStocks.map((s, i) => (
                  <span key={s.stockCode}>
                    {i > 0 && "、"}
                    <Link href={`/stocks/${s.stockCode}`} className="font-bold text-brand-blue hover:underline">
                      {s.stockName}（{s.stockCode}）
                    </Link>
                    （{s.count}件・{formatDealAmount(s.amount)}）
                  </span>
                ))}
                への開示が目立ちました。
              </>
            )}
          </p>
        </section>
      )}

      {featured.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-4 text-lg font-bold text-brand-navy">注目の取引</h2>
          <div className="space-y-4">
            {featured.map((article, i) => (
              <FeaturedArticleCard key={article.id} article={article} rank={i + 1} />
            ))}
          </div>
        </section>
      )}

      {dateGroups.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-bold text-brand-navy">日別の記事一覧</h2>
          <div className="divide-y divide-rule border-y border-rule">
            {dateGroups.map((group) => {
              const dayAmount = group.articles.reduce((sum, a) => sum + a.dealAmount, 0);
              return (
                <Link
                  key={group.date}
                  href={`/date/${group.date}`}
                  className="group flex items-center justify-between gap-4 py-4 transition-colors hover:bg-section-tint"
                >
                  <div>
                    <p className="font-bold text-brand-navy">{group.label}</p>
                    <p className="mt-0.5 text-sm text-foreground/60">
                      {group.articles.length}件・{formatDealAmount(dayAmount)}
                    </p>
                  </div>
                  <span className="kicker shrink-0 text-brand-blue transition-colors group-hover:text-brand-navy">
                    この日の記事を見る ›
                  </span>
                </Link>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

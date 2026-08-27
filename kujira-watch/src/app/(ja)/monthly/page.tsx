import type { Metadata } from "next";
import Link from "next/link";
import MonthList from "@/components/MonthList";
import { siblingDataPages } from "@/lib/nav";
import ListPageNextStep from "@/components/ListPageNextStep";
import RelatedArticles from "@/components/RelatedArticles";
import { formatMonth } from "@/lib/format";
import { getAllMonthsForIndex, getArticleList } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";

export const revalidate = 60;

const url = `${SITE_URL}/monthly`;
const title = "月別アーカイブ";
// H1・パンくずは短いラベル（title）のまま、検索結果に出す<title>だけ検索語を入れた形にする。
// GA4の実測（28日）でデータ/一覧ページは940PVのうち889＝95%が内部到達で、入口はわずか51。
// 滞在75秒と全種別で最も長いのに検索から直接来ていない。説明文には既に検索語が入っている
// 一方で<title>が「銘柄ランキング」のような内部呼称のままだったため、そこを揃える（2026-08-27）。
// ※SEOの反映には数日〜数週間かかるので、直後に順位で判定しないこと。
const metaTitle = "大量保有報告書の月別アーカイブ";

export const metadata: Metadata = {
  title: metaTitle,
  description:
    "EDINET大量保有報告書をもとにした大口投資家の動きを月ごとにまとめたアーカイブ。各月の開示件数・推定取引金額から、その月に動いた投資家・銘柄をたどれます。",
  alternates: { canonical: url },
  openGraph: { title: metaTitle, url },
};

// 月別アーカイブの入口。取引日別ページ(/date/[date])は日数分だけ増える一方で、
// 直近7日ぶんが/weeklyから張られるだけで古い日付は内部リンクが切れていた
// （サイトマップにしか載らない孤立ページになっていた）。月ハブを挟むことで
// 全ての取引日別ページが「ヘッダー → 月別アーカイブ → 各月 → 各日」で辿れるようにする。
export default async function MonthlyIndexPage() {
  const [months, { contents: latestArticles }] = await Promise.all([
    getAllMonthsForIndex(),
    // 月リストの下に添えるアイキャッチ付き記事カード用。取れなくても一覧は成立させる。
    getArticleList({ limit: 4 }).catch(() => ({ contents: [] })),
  ]);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: title,
    itemListElement: months.map((m, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: `${formatMonth(m.month)}の大口投資家の動き`,
      url: `${SITE_URL}/monthly/${m.month}`,
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
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">月別アーカイブ</h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}が公開した大口投資家の動きを月ごとにまとめています。
          各月のページでは、その月に動いた投資家・銘柄のランキングと日別の一覧を確認できます。
        </p>
      </div>
      {months.length === 0 ? (
        <p className="text-sm text-foreground/60">まだ記事がありません。</p>
      ) : (
        <MonthList months={months} />
      )}
      <div className="mt-10">
        <RelatedArticles
          title="最新の解説記事"
          lead="今月ぶんのアーカイブに入る、直近の取引の解説記事です。"
          articles={latestArticles}
        />
      </div>
      {/* データページ同士の横移動。ヘッダータブはあるが、GA4実測でTOPへの内部到達398件＝
          他ページからTOPへ戻る動きが多く、横に渡り歩けていなかった（2026-08-27）。 */}
      <ListPageNextStep links={siblingDataPages("/monthly")} />
      <AdUnit placement="bottom" />
    </div>
  );
}

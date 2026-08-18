import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import Box from "@mui/material/Box";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import Typography from "@mui/material/Typography";
import PriceAfterDisclosure from "@/components/PriceAfterDisclosure";
import CategoryBadge from "@/components/CategoryBadge";
import DealDirectionBadge from "@/components/DealDirectionBadge";
import ActionButton from "@/components/ActionButton";
import ArticleCard from "@/components/ArticleCard";
import FollowCta from "@/components/FollowCta";
import ShareButtons from "@/components/ShareButtons";
import { displayFilerName, excerptFromHtml, formatDate, formatDealAmount, linkifyFilerNames } from "@/lib/format";
import { getArticleDetail, getArticleList, getArticlesByStockCode } from "@/lib/microcms";
import { getFilerNamesByStockAndDate, getFilersByStockCode, getHoldingSnapshot } from "@/lib/investors";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { isIndexableArticle } from "@/lib/articleIndexability";
import { categoryLabel } from "@/types/article";
import type { ArticleContent } from "@/types/article";
import AdUnit from "@/components/AdUnit";

// Route segment config requires a literal value (cannot import from lib/microcms).
// 記事本文は公開後ほぼ変わらない（変わるのは株価チャート等の子コンポーネント側）。
// 60秒だとクローラーのアクセスがほぼ毎回キャッシュ切れに当たって再生成となり、
// 実測TTFBが1.8〜2.8秒まで悪化していたため1日に延ばす。記事をリライトした場合は
// 最大1日反映が遅れるが、リライト自体が稀なので許容する。
export const revalidate = 86400;

// 「同じ銘柄」「同じ投資家」の関連リンクの表示件数。カード表示の関連記事（同じ分類）と
// 違って一行リンクで並べるため、多すぎない範囲で回遊先を増やす。
const RELATED_COUNT = 3;

// 関連リンクの一行表示（取引日・金額つき）。回遊導線なのでカードより密度を優先する。
function RelatedArticleLinks({ articles }: { articles: ArticleContent[] }) {
  return (
    <List disablePadding sx={{ borderTop: 1, borderBottom: 1, borderColor: "divider" }}>
      {articles.map((related) => (
        <ListItem key={related.id} disableGutters divider sx={{ py: 1.5 }}>
          <Link href={`/articles/${related.id}`} className="group flex w-full items-baseline justify-between gap-4">
            <Typography
              component="span"
              variant="body2"
              className="group-hover:underline"
              sx={{ fontWeight: 500, color: "primary.main", ".group:hover &": { color: "brand.blue" } }}
            >
              {related.title}
            </Typography>
            <Typography component="span" variant="caption" sx={{ flexShrink: 0, color: "text.disabled" }}>
              {formatDate(related.dealDate)}・{formatDealAmount(related.dealAmount)}
            </Typography>
          </Link>
        </ListItem>
      ))}
    </List>
  );
}

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const article = await getArticleDetail(id).catch(() => null);
  if (!article) return {};

  const description = `${article.stockName}(${article.stockCode})の${article.dealType}を解説。${excerptFromHtml(article.body)}`;
  const url = `${SITE_URL}/articles/${id}`;
  const hasEn = Boolean(article.titleEn && article.bodyEn);

  return {
    title: article.title,
    description,
    // 金額も保有比率の変化も小さい開示（例: 保有比率0.04%・推定額0億円の変更報告書）は
    // 検索意図を満たさず、テンプレート全体の評価を下げるためインデックスさせない。
    // followは残すので、この記事から銘柄・投資家ページへのリンクはクロールされる。
    ...(isIndexableArticle(article) ? {} : { robots: { index: false, follow: true } }),
    alternates: {
      canonical: url,
      ...(hasEn ? { languages: { ja: url, en: `${SITE_URL}/en/articles/${id}` } } : {}),
    },
    openGraph: {
      type: "article",
      url,
      title: article.title,
      description,
      // 記事が外に出す日付は「EDINET開示日(dealDate)」に一本化する。microCMSのupdatedAtは
      // 分類の付け替えやフィールド削除などの一括バッチでも動いてしまい、「全記事が昨日更新」に
      // なって更新日の信号として死ぬ。publishedAtも過去分の一括生成（バックフィル）で
      // 生成日に寄るため、記事が扱う事実の日付とずれる。sitemapの<lastmod>も同じ基準。
      publishedTime: article.dealDate,
      modifiedTime: article.dealDate,
      ...(article.eyecatch ? { images: [article.eyecatch.url] } : {}),
    },
    twitter: {
      card: "summary_large_image",
      title: article.title,
      description,
    },
  };
}

export default async function ArticleDetailPage({ params }: Props) {
  const { id } = await params;

  const article = await getArticleDetail(id).catch((error: unknown) => {
    if (error instanceof Error && error.message.includes("status: 404")) {
      notFound();
    }
    throw error;
  });

  const tags = article.tags
    ?.split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);

  const category = article.dealType ? categoryLabel(article.dealType) : undefined;
  const url = `${SITE_URL}/articles/${id}`;
  const dealDateOnly = article.dealDate.slice(0, 10);

  // 関連リンクは「同じ銘柄」→「同じ分類」の順に近いものから並べる。
  // 同じ記事・既に上のセクションで出した記事は重複させない。
  const [{ contents: sameCategoryArticles }, { contents: sameStockArticles }, filers, filerByKey] =
    await Promise.all([
      article.dealType ? getArticleList({ dealType: article.dealType, limit: 5 }) : Promise.resolve({ contents: [] }),
      getArticlesByStockCode(article.stockCode),
      getFilersByStockCode(article.stockCode),
      getFilerNamesByStockAndDate(dealDateOnly, dealDateOnly),
    ]);

  const relatedStockArticles = sameStockArticles.filter((a) => a.id !== id).slice(0, RELATED_COUNT);
  const shownIds = new Set([id, ...relatedStockArticles.map((a) => a.id)]);
  const relatedArticles = sameCategoryArticles.filter((a) => !shownIds.has(a.id)).slice(0, 4);

  // microCMSのarticlesスキーマには提出者名フィールドが無く、article.filerNameは常にundefinedに
  // なる（lib/investors.tsのgetFilerNamesByStockAndDateの注記を参照）。CMS側にフィールドが
  // 追加されればそれを優先し、無い間はEDINET開示（Supabase）との突き合わせで補う。
  const filerName =
    article.filerName ?? filerByKey.get(`${article.stockCode}|${dealDateOnly}`);

  // ファクトボックス用: 保有比率はCMSに無いため、提出者が特定できた記事のみEDINET開示から引く。
  const snapshot = filerName
    ? await getHoldingSnapshot(article.stockCode, dealDateOnly, filerName)
    : null;
  const holdingRatio = snapshot?.holdingRatio ?? null;
  // 前回比はCMSのratioChangePct（2026-08-15以降の記事）を優先し、
  // 無ければEDINET開示の直前保有割合との差から補う。
  const ratioChange =
    article.ratioChangePct ??
    (holdingRatio !== null && snapshot?.holdingRatioPrior != null
      ? Math.round((holdingRatio - snapshot.holdingRatioPrior) * 100) / 100
      : null);
  const formatRatio = (value: number) =>
    value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");

  const linkedBody = linkifyFilerNames(article.body, filers.map((f) => f.filerName));

  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: article.title,
    url,
    // 日付はEDINET開示日に一本化（generateMetadataのpublishedTimeの注記を参照）。
    // 画面に出している「開示日」とも一致させ、構造化データと可視コンテンツをずらさない。
    datePublished: article.dealDate,
    dateModified: article.dealDate,
    inLanguage: "ja",
    mainEntityOfPage: { "@type": "WebPage", "@id": url },
    about: {
      "@type": "Corporation",
      name: article.stockName,
      tickerSymbol: article.stockCode,
    },
    articleSection: category,
    // 記事は人手ではなくAIがEDINET開示等の事実情報から生成しているため、
    // authorはサイト運営組織そのもの（Organization）とする。
    author: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
    publisher: {
      "@type": "Organization",
      name: SITE_NAME,
      url: SITE_URL,
      logo: { "@type": "ImageObject", url: `${SITE_URL}/logo` },
    },
    ...(article.eyecatch ? { image: article.eyecatch.url } : {}),
    ...(article.sourceUrl ? { citation: article.sourceUrl } : {}),
  };

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      {
        "@type": "ListItem",
        position: 2,
        name: formatDate(article.dealDate),
        item: `${SITE_URL}/date/${dealDateOnly}`,
      },
      { "@type": "ListItem", position: 3, name: article.title, item: url },
    ],
  };

  return (
    <article className="overflow-hidden bg-paper">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      {/* 記事タイトルは長く、そのまま置くとパンくずが3行になり本文到達前のノイズに
          なるため1行ellipsisで切る。SEO用のBreadcrumbList(JSON-LD)はフルタイトルのまま。 */}
      <nav aria-label="パンくずリスト" className="flex items-center gap-1.5 border-b border-rule px-6 py-3 text-xs text-foreground/50">
        <Link href="/" className="flex-none hover:text-brand-blue">トップ</Link>
        <span aria-hidden>/</span>
        <Link href={`/date/${dealDateOnly}`} className="flex-none hover:text-brand-blue">
          {formatDate(article.dealDate)}
        </Link>
        <span aria-hidden>/</span>
        <span className="min-w-0 truncate text-foreground/70">{article.title}</span>
      </nav>
      {article.eyecatch && (
        <div className="relative aspect-video w-full bg-section-tint">
          <Image
            src={article.eyecatch.url}
            alt={article.eyecatch.alt || article.title}
            fill
            priority
            className="object-cover"
            sizes="(min-width: 768px) 768px, 100vw"
          />
        </div>
      )}
      <div className="p-6 sm:p-10">
        <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          {/* 分類はリンク付きChip(CategoryBadge)へ一本化。ドット版(DealTypeBadge)を併記すると
              同一ラベルが二重表示になっていた。 */}
          <CategoryBadge dealType={article.dealType} />
          <DealDirectionBadge tags={article.tags} />
        </div>
        <h1 className="mb-4 text-2xl font-bold leading-snug text-brand-navy sm:text-3xl">
          {article.title}
        </h1>
        <Box
          component="dl"
          sx={{
            m: 0,
            mb: 4,
            py: 2,
            borderTop: 1,
            borderBottom: 1,
            borderColor: "divider",
            display: "grid",
            gridTemplateColumns: { xs: "repeat(2, 1fr)", sm: "repeat(4, 1fr)" },
            columnGap: 2,
            rowGap: 2,
          }}
        >
          {/* 銘柄名・投資家名は折り返し不能な長い固有名詞のため、モバイルの半カラム幅では
              1行3〜4文字の縦割れになる。この2項目だけはxsで行全体を使う。 */}
          <Box sx={{ gridColumn: { xs: "1 / -1", sm: "auto" } }}>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>銘柄</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
              <Link
                href={`/stocks/${article.stockCode}`}
                className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
              >
                {article.stockName}（{article.stockCode}）
              </Link>
            </Typography>
          </Box>
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>取引日</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {formatDate(article.dealDate)}
            </Typography>
          </Box>
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>金額規模</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {formatDealAmount(article.dealAmount)}
            </Typography>
          </Box>
          {holdingRatio !== null && (
            <Box>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>保有比率</Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
                {formatRatio(holdingRatio)}%
                {snapshot?.holdingRatioPrior != null && (
                  <Typography component="span" variant="caption" sx={{ ml: 0.5, color: "text.disabled" }}>
                    （前回 {formatRatio(snapshot.holdingRatioPrior)}%）
                  </Typography>
                )}
              </Typography>
            </Box>
          )}
          {ratioChange !== null && ratioChange !== 0 && (
            <Box>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>前回比</Typography>
              <Typography
                component="dd"
                sx={{ m: 0, mt: 0.5, fontWeight: 500, color: ratioChange > 0 ? "success.main" : "error.main" }}
              >
                {ratioChange > 0 ? "＋" : "−"}{formatRatio(Math.abs(ratioChange))}pt
              </Typography>
            </Box>
          )}
          {filerName && (
            <Box sx={{ gridColumn: { xs: "1 / -1", sm: "auto" } }}>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>取引企業</Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
                <Link
                  href={`/investors/${encodeURIComponent(filerName)}`}
                  className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
                >
                  {displayFilerName(filerName)}
                </Link>
              </Typography>
            </Box>
          )}
          {article.sourceUrl && (
            <Box>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>出典</Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5 }}>
                <a
                  href={article.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-brand-blue hover:underline"
                >
                  元記事を見る
                </a>
              </Typography>
            </Box>
          )}
        </Box>
        {/* 全記事で同一の説明文（提出期限のズレ・免責・分類の定義）はここに書かず、
            FAQと/aboutへのリンクに寄せる。同じ定型文が全記事の本文比率を押し上げると
            「他ページと内容が酷似」と判定されるため（GSCのクロール済み-未登録の主因）。 */}
        <p className="-mt-2 mb-6 text-xs text-foreground/40">
          ※ 取引日はEDINETの開示日です（
          <Link href="/faq/basics" className="text-brand-blue hover:underline">
            実際の売買とのずれ
          </Link>
          ・
          <Link href="/about" className="text-brand-blue hover:underline">
            免責
          </Link>
          ）
        </p>
        <PriceAfterDisclosure stockCode={article.stockCode} dealDate={article.dealDate} />
        <div
          className="prose max-w-none prose-headings:text-brand-navy prose-a:text-brand-blue first:prose-p:first-letter:float-left first:prose-p:first-letter:mr-2 first:prose-p:first-letter:text-5xl first:prose-p:first-letter:font-bold first:prose-p:first-letter:text-brand-navy"
          dangerouslySetInnerHTML={{ __html: linkedBody }}
        />
        <div className="mt-8 rounded border border-rule bg-section-tint px-4 py-3 text-xs leading-relaxed text-foreground/60">
          <p className="m-0">
            情報源: 金融庁EDINETの大量保有報告書等（提出日: {formatDate(article.dealDate)}）
            {article.sourceUrl && (
              <>
                {" ・ "}
                <a
                  href={article.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-brand-blue hover:underline"
                >
                  元の開示を見る
                </a>
              </>
            )}
          </p>
          <p className="m-0 mt-1">金額は発行済株式数と株価からの概算です。</p>
        </div>
        {tags && tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-x-3 gap-y-1 border-t border-rule pt-4 text-xs text-foreground/50">
            {tags.map((tag) => (
              <span key={tag}>#{tag}</span>
            ))}
          </div>
        )}
        <ShareButtons url={url} title={article.title} />
        <FollowCta />
        {relatedStockArticles.length > 0 && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-4 text-xl font-bold text-brand-navy">
              {article.stockName}（{article.stockCode}）の他の記事
            </h2>
            <RelatedArticleLinks articles={relatedStockArticles} />
            <div className="mt-3">
              <ActionButton href={`/stocks/${article.stockCode}`}>
                この銘柄の大量保有・自社株買い履歴をすべて見る
              </ActionButton>
            </div>
          </div>
        )}
        {filerName && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-2 text-xl font-bold text-brand-navy">この取引をした投資家</h2>
            <ActionButton href={`/investors/${encodeURIComponent(filerName)}`}>
              {displayFilerName(filerName)}の保有銘柄・比率推移を見る
            </ActionButton>
          </div>
        )}
        <div className="mt-10 border-t border-rule pt-6">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">関連ランキング</h2>
          <nav aria-label="関連ランキング" className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
            <Link href="/ranking/buys" className="text-brand-blue hover:underline">買い増しランキング</Link>
            <Link href="/ranking/sells" className="text-brand-blue hover:underline">売却ランキング</Link>
            <Link href={`/date/${dealDateOnly}`} className="text-brand-blue hover:underline">
              {formatDate(article.dealDate)}の全開示
            </Link>
          </nav>
        </div>
        {relatedArticles.length > 0 && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-5 text-xl font-bold text-brand-navy">
              関連記事（{category}）
            </h2>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              {relatedArticles.map((related) => (
                <ArticleCard key={related.id} article={related} />
              ))}
            </div>
          </div>
        )}
      </div>
      <AdUnit placement="bottom" />
    </article>
  );
}

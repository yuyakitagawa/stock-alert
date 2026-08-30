import type { Metadata } from "next";
import { notFound, permanentRedirect } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import Box from "@mui/material/Box";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import Typography from "@mui/material/Typography";
import PriceAfterDisclosure from "@/components/PriceAfterDisclosure";
import ArticleNextStep from "@/components/ArticleNextStep";
import HoldingRatioChart from "@/components/HoldingRatioChart";
import HoldingPurposeBadge from "@/components/HoldingPurposeBadge";
import CategoryBadge from "@/components/CategoryBadge";
import DealDirectionBadge from "@/components/DealDirectionBadge";
import ActionButton from "@/components/ActionButton";
import ArticleCard from "@/components/ArticleCard";
import FollowCta from "@/components/FollowCta";
import ShareButtons from "@/components/ShareButtons";
import { displayFilerName, excerptFromHtml, formatDate, formatDealAmount, formatDealAmountOrCorrection, frameSpeculation, isCorrectionArticle, linkifyFilerNames } from "@/lib/format";
import {
  getAllArticlesForSitemap,
  getArticleDetail,
  getArticleList,
  getArticlesByStockCode,
} from "@/lib/microcms";
import { getFilerIdByName, getFilerNamesByStockAndDate, getFilersByStockCode, getHoldingHistory, getHoldingSnapshot, investorPath } from "@/lib/investors";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { averageAcquisitionPrice, borrowingRatio, classifyPurpose, filingLagDays } from "@/lib/disclosures";
import { SITE_NAME, SITE_URL, X_HANDLE } from "@/lib/site";
import { isIndexableArticle, supersededArticleIds } from "@/lib/articleIndexability";
import { resolveArticleRedirect } from "@/lib/articleRedirects";
import { dateHref, getPublishedFilerNames, stockHref } from "@/lib/publishedPages";
import { categoryLabel } from "@/types/article";
import type { ArticleContent } from "@/types/article";
import AdUnit from "@/components/AdUnit";

// Route segment config requires a literal value (cannot import from lib/microcms).
// 記事本文は公開後ほぼ変わらない（変わるのは株価チャート等の子コンポーネント側）。
// 60秒だとクローラーのアクセスがほぼ毎回キャッシュ切れに当たって再生成となり、
// 実測TTFBが1.8〜2.8秒まで悪化していたため1日に延ばす。記事をリライトした場合は
// 最大1日反映が遅れるが、リライト自体が稀なので許容する。
export const revalidate = 86400;

// 直近の記事はビルド時に静的生成する。Next 16では generateStaticParams の無い動的セグメントは
// リクエストごとのSSR（実測: x-vercel-cache: MISS / cache-control: no-store）になり、
// クローラーが同じURLを取りに来るたびにサーバー実行になる。新着記事ほどクロールされる頻度が
// 高いので、直近分だけ事前生成してCDNから即返す（それ以前の記事は従来どおり都度レンダリング）。
const PRERENDERED_ARTICLES = 200;

export async function generateStaticParams() {
  // 一覧の取得に失敗しても空配列を返してビルドは通す（microCMS/Supabaseの一時障害で
  // デプロイ全体を落とさないため）。空でもルートはISR扱いのままになる。
  try {
    const articles = await getAllArticlesForSitemap();
    return articles.slice(0, PRERENDERED_ARTICLES).map((article) => ({ id: article.id }));
  } catch {
    return [];
  }
}

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
              {formatDate(related.dealDate)}・{formatDealAmountOrCorrection(related)}
            </Typography>
          </Link>
        </ListItem>
      ))}
    </List>
  );
}

// 同一銘柄の記事一覧（最大100件）で判定する。サイトマップ側は全記事で同じ関数を通しており、
// 1銘柄あたりの記事が100件を超えない限り両者の判定は一致する（超えた場合はindex側に倒れる）。
async function isSupersededArticle(id: string, stockCode: string | undefined): Promise<boolean> {
  if (!stockCode) return false;
  const { contents } = await getArticlesByStockCode(stockCode);
  return supersededArticleIds(contents).has(id);
}

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const article = await getArticleDetail(id).catch(() => null);
  if (!article) return {};

  // dealTypeは投資家分類（例: 日系証券銀行）なので「横河電機の日系証券銀行を解説」とは
  // 日本語として繋がらない。検索結果の説明文は「銘柄｜投資家分類の大量保有報告書を解説。」に
  // 揃え、続く本文1文目（直答文）で誰が何%にしたかを伝える。
  const description =
    article.dealType === "自社株買い"
      ? `${article.stockName}（${article.stockCode}）｜自社株買い（TDnet適時開示）の取得枠を解説。${excerptFromHtml(article.body)}`
      : `${article.stockName}（${article.stockCode}）｜${article.dealType}の大量保有報告書を解説。${excerptFromHtml(article.body)}`;
  const url = `${SITE_URL}/articles/${id}`;
  // 同一「銘柄×提出者」の記事は最新1本だけをindexする（カニバリゼーション対策。
  // 詳細はlib/articleIndexability.tsのsupersededArticleIds()を参照）。
  const superseded = await isSupersededArticle(id, article.stockCode);
  const indexable = isIndexableArticle(article) && !superseded;

  return {
    // サイト名サフィックス（｜大口投資家の監視ブログ＝全角12字）を付けない。記事タイトルは
    // 銘柄名・提出者名・保有比率だけで既に40字前後あり、検索結果に出る約32字にサイト名は
    // 入らない。入る場合も本文側の情報を押し出すだけで、人名・銘柄名で探している読者には
    // 何の手がかりにもならない。一覧・ハブページのtitleにはサイト名を残す。
    title: { absolute: article.title },
    description,
    // 金額も保有比率の変化も小さい開示（例: 保有比率0.04%・推定額0億円の変更報告書）と、
    // 同一「銘柄×提出者」で最新に置き換わった記事は、検索意図を満たさず
    // テンプレート全体の評価を下げるためインデックスさせない。
    // followは残すので、この記事から銘柄・投資家ページへのリンクはクロールされる。
    ...(indexable ? {} : { robots: { index: false, follow: true } }),
    alternates: {
      canonical: url,
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
      site: X_HANDLE,
      creator: X_HANDLE,
      title: article.title,
      description,
    },
  };
}

export default async function ArticleDetailPage({ params }: Props) {
  const { id } = await params;

  const article = await getArticleDetail(id).catch((error: unknown) => {
    if (error instanceof Error && error.message.includes("status: 404")) {
      return null;
    }
    throw error;
  });

  // 削除済みの記事URLは404にせず、引き継ぎ先（銘柄ページ／重複で残した方の記事）へ
  // 恒久リダイレクトする。順位の付いたURLを404で捨てないための処理。
  // 引き継ぎ先が非公開になっている場合はnullが返るので、そのときは素直に404にする
  // （308→404の二段は素の404より悪い）。
  if (!article) {
    const target = await resolveArticleRedirect(id);
    if (target) permanentRedirect(target);
    notFound();
  }

  // 「自動生成」は運用側の内部フラグ（web/publish_blog_articles.pyがtagsに立てる）で、
  // 読者に見せると記事の信頼性を落とすだけなので表示しない。方向・訂正のタグは残す。
  const HIDDEN_TAGS = ["自動生成"];
  const tags = article.tags
    ?.split(",")
    .map((tag) => tag.trim())
    .filter((tag) => Boolean(tag) && !HIDDEN_TAGS.includes(tag));

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
  // 自社株買い記事は提出者が発行体自身で、同日に同銘柄の大量保有報告書があっても無関係なので
  // EDINET突き合わせはしない（投資家名・保有比率ブロックはすべて出さない）。
  const isBuyback = article.dealType === "自社株買い";
  const filerName = isBuyback
    ? undefined
    : article.filerName ?? filerByKey.get(`${article.stockCode}|${dealDateOnly}`);

  // ファクトボックス用: 保有比率はCMSに無いため、提出者が特定できた記事のみEDINET開示から引く。
  const snapshot = filerName
    ? await getHoldingSnapshot(article.stockCode, dealDateOnly, filerName)
    : null;
  const holdingRatio = snapshot?.holdingRatio ?? null;
  // 短期大量譲渡の開示（法第27条の25第2項）だけは「誰にいくらで売ったか」が原文に載る。
  // EDINETが金額を出さないこのサイトで、概算でない実額を出せる唯一のケース。
  const transfers = snapshot?.transfers ?? null;
  const exactAmountOku = transfers?.amountOku ?? null;
  // 大量保有報告書XBRLの本表から取れる、EDINETの原文にしか載っていない事実。
  // 保有目的はほぼ全開示にあるが、取得資金・保有株数は記載の無い開示や
  // 全部売却した変更報告書では欠けるので、それぞれ取れたものだけ出す。
  const purpose = classifyPurpose(snapshot?.purposeOfHolding ?? null);
  const unitPrice = averageAcquisitionPrice(snapshot?.fundingTotal ?? null, snapshot?.sharesHeld ?? null);
  const leverage = borrowingRatio(snapshot?.fundingBorrowings ?? null, snapshot?.fundingTotal ?? null);
  const filingLag = filingLagDays(snapshot?.obligationDate ?? null, dealDateOnly);
  // 法定は報告義務発生日から5営業日以内。1ヶ月を超える遅れは「今さら出てきた開示」で、
  // 株価が既に動いた後という読み方が要るため、そのときだけ出す。
  const isLateFiling = filingLag !== null && filingLag > 30;
  // formatDealAmountは億円単位を受ける。0.05億円未満は「0億円」になって金額が無いように
  // 見えるので出さない（短期大量譲渡の実額と同じ足切り）。
  const fundingOku =
    snapshot?.fundingTotal && snapshot.fundingTotal >= 5e6
      ? Math.round((snapshot.fundingTotal / 1e8) * 10) / 10
      : null;
  // 同一投資家×同一銘柄の開示履歴（2件以上あるときだけチャートを描く）
  const holdingHistory = filerName ? await getHoldingHistory(article.stockCode, filerName) : [];
  const filerId = filerName ? await getFilerIdByName(filerName) : null;
  // 薄い集約ページは公開していない（404）ので、リンクも同じ判定で出し分ける。
  // nullならリンクにせず素のテキストで出す（lib/publishedPages.ts）。
  const [stockPageHref, datePageHref, publishedFilers] = await Promise.all([
    stockHref(article.stockCode),
    dateHref(dealDateOnly),
    getPublishedFilerNames(),
  ]);
  const filerPageHref =
    filerName && publishedFilers.has(filerName) ? investorPath(filerId, filerName) : null;
  // 前回比はEDINET開示（今回比率 − 直前保有割合）を正とし、取れないときだけCMSの
  // ratioChangePctへフォールバックする。CMS側の値は記事生成時にXBRLの直前保有割合が
  // まだ取れていないと「今回比率の全量」で入ってしまい、同じ画面に
  // 「保有比率 10.57%（前回 10.72%）」と「前回比 −10.57pt」が並ぶ矛盾が起きていた
  // （2026-08-19の監査で検出。生成側は publish_blog_articles.should_wait_for_prior_ratio で対処済み）。
  const ratioChange =
    holdingRatio !== null && snapshot?.holdingRatioPrior != null
      ? Math.round((holdingRatio - snapshot.holdingRatioPrior) * 100) / 100
      : article.ratioChangePct ?? null;
  const formatRatio = (value: number) =>
    value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");

  const linkedBody = frameSpeculation(
    linkifyFilerNames(
      article.body,
      filers.filter((f) => publishedFilers.has(f.filerName))
    )
  );

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
      // 取引日ページを公開していない日は階層から外す（存在しないURLを構造化データに出さない）。
      ...(datePageHref
        ? [
            {
              "@type": "ListItem",
              position: 2,
              name: formatDate(article.dealDate),
              item: `${SITE_URL}${datePageHref}`,
            },
          ]
        : []),
      { "@type": "ListItem", position: datePageHref ? 3 : 2, name: article.title, item: url },
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
      <nav aria-label="パンくずリスト" className="flex items-center gap-1.5 border-b border-rule px-6 py-3 text-xs text-ink-tertiary">
        <Link href="/" className="flex-none hover:text-brand-blue">トップ</Link>
        <span aria-hidden>/</span>
        {datePageHref ? (
          <Link href={datePageHref} className="flex-none hover:text-brand-blue">
            {formatDate(article.dealDate)}
          </Link>
        ) : (
          <span className="flex-none">{formatDate(article.dealDate)}</span>
        )}
        <span aria-hidden>/</span>
        <span className="min-w-0 truncate text-ink-secondary">{article.title}</span>
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
              {stockPageHref ? (
                <Link
                  href={stockPageHref}
                  className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
                >
                  {article.stockName}（{article.stockCode}）
                </Link>
              ) : (
                <>
                  {article.stockName}（{article.stockCode}）
                </>
              )}
            </Typography>
          </Box>
          <Box>
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>取引日</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {formatDate(article.dealDate)}
            </Typography>
          </Box>
          <Box>
            {/* EDINETは保有"比率"しか開示せず、金額は発行済株式数×株価×比率変化の概算。
                桁が大きいほど断定的に見えるため、数字のすぐ隣に「概算」と添える。 */}
            <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>金額規模</Typography>
            <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500, color: "primary.main" }}>
              {exactAmountOku !== null && !isCorrectionArticle(article.tags)
                ? formatDealAmount(exactAmountOku)
                : formatDealAmountOrCorrection(article)}
              {!isCorrectionArticle(article.tags) && (
                <Typography component="span" variant="caption" sx={{ ml: 0.5, color: "text.disabled" }}>
                  {/* 自社株買いは取締役会決議の取得枠上限（概算ではなく開示値）。
                      短期大量譲渡は開示された譲渡単価×株数の実額なので概算と区別する。 */}
                  {isBuyback ? "（上限）" : exactAmountOku !== null ? "（開示単価ベース）" : "（概算）"}
                </Typography>
              )}
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
              {/* 自社株買い記事の ratioChangePct には取得枠の発行済株式比率（上限）を入れている */}
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
                {isBuyback ? "発行済比率（上限）" : "前回比"}
              </Typography>
              <Typography
                component="dd"
                sx={{ m: 0, mt: 0.5, fontWeight: 500, color: ratioChange > 0 ? "success.main" : "error.main" }}
              >
                {isBuyback
                  ? `${formatRatio(ratioChange)}%`
                  : `${ratioChange > 0 ? "＋" : "−"}${formatRatio(Math.abs(ratioChange))}pt`}
              </Typography>
            </Box>
          )}
          {filerName && (
            <Box sx={{ gridColumn: { xs: "1 / -1", sm: "auto" } }}>
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>取引企業</Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
                <Link
                  href={investorPath(filerId, filerName)}
                  className="text-brand-blue underline decoration-brand-blue/40 underline-offset-2 hover:decoration-brand-blue"
                >
                  {displayFilerName(filerName)}
                </Link>
              </Typography>
            </Box>
          )}
          {snapshot?.purposeOfHolding && (
            <Box sx={{ gridColumn: "1 / -1" }}>
              {/* 保有目的は大量保有報告書の必須記載項目で、EDINETのXBRLからほぼ全開示で取れる。
                  同じ「5%取得」でも純投資と重要提案行為等では意味が違うため、分類バッジと
                  原文を併記する（分類はあくまで自由記述からの機械判定）。 */}
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
                保有目的（開示原文より）
              </Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5 }}>
                {purpose && (
                  <Box component="span" sx={{ mr: 0.75, verticalAlign: "middle" }}>
                    <HoldingPurposeBadge purpose={purpose} />
                  </Box>
                )}
                {/* 原文が「純投資」だけの開示ではバッジと同じ文字が2つ並ぶので原文側を省く。 */}
                {snapshot.purposeOfHolding !== purpose && (
                  <Typography component="span" variant="body2" sx={{ whiteSpace: "pre-line" }}>
                    {snapshot.purposeOfHolding}
                  </Typography>
                )}
              </Typography>
            </Box>
          )}
          {snapshot?.importantProposal && !/該当(事項)?(なし|ありません)/.test(snapshot.importantProposal) && (
            <Box sx={{ gridColumn: "1 / -1" }}>
              {/* 「重要提案行為等」欄に具体的な記載がある開示だけ出す（大半は「該当事項なし」）。 */}
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
                重要提案行為等（開示原文より）
              </Typography>
              <Typography component="dd" variant="body2" sx={{ m: 0, mt: 0.5, whiteSpace: "pre-line" }}>
                {snapshot.importantProposal}
              </Typography>
            </Box>
          )}
          {unitPrice !== null && (
            <Box>
              {/* 取得資金の総額÷保有株数。EDINETは通常「比率」しか出さないが、取得資金は
                  本表に金額で載っているので取得原価が出せる。保有が古いほど現在株価から
                  離れる（政策保有株は特に）ため参考値として扱う。 */}
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
                平均取得単価（開示ベース）
              </Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
                {unitPrice.toLocaleString("ja-JP")}円
                {snapshot?.sharesHeld ? (
                  <Typography component="span" variant="caption" sx={{ ml: 0.5, color: "text.disabled" }}>
                    （{snapshot.sharesHeld.toLocaleString("ja-JP")}株）
                  </Typography>
                ) : null}
              </Typography>
            </Box>
          )}
          {leverage !== null && leverage > 0 && (
            <Box>
              {/* 取得資金に占める借入金の割合。自己資金0＝全額借入の買いが実在し、
                  返済圧力がある分だけ同じ保有比率でも意味が違う。 */}
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
                借入比率
              </Typography>
              <Typography
                component="dd"
                sx={{ m: 0, mt: 0.5, fontWeight: 500, color: leverage >= 50 ? "warning.main" : undefined }}
              >
                {leverage}%
                {fundingOku !== null && (
                  <Typography component="span" variant="caption" sx={{ ml: 0.5, color: "text.disabled" }}>
                    （取得資金{formatDealAmount(fundingOku)}のうち借入）
                  </Typography>
                )}
              </Typography>
            </Box>
          )}
          {isLateFiling && snapshot?.obligationDate && (
            <Box>
              {/* 法定は報告義務発生日から5営業日以内。1ヶ月超の遅れは「株価が動いた後に
                  出てきた開示」なので、そのときだけ出す。 */}
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
                報告義務発生日
              </Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
                {formatDate(snapshot.obligationDate)}
                <Typography component="span" variant="caption" sx={{ ml: 0.5, color: "text.disabled" }}>
                  （提出まで{filingLag}日）
                </Typography>
              </Typography>
            </Box>
          )}
          {transfers && transfers.counterparties.length > 0 && (
            <Box sx={{ gridColumn: "1 / -1" }}>
              {/* 短期大量譲渡の原文に載る相手方。通常の大量保有報告書には無い情報なので、
                  取れた記事だけに出す（この列があるのは全開示の3%弱）。 */}
              <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
                譲渡の相手方（開示原文より）
              </Typography>
              <Typography component="dd" sx={{ m: 0, mt: 0.5, fontWeight: 500 }}>
                {transfers.counterparties.slice(0, 4).join("、")}
                {transfers.counterparties.length > 4 && `ほか${transfers.counterparties.length - 4}者`}
                {transfers.unitPrice !== null && (
                  <Typography component="span" variant="caption" sx={{ ml: 0.5, color: "text.disabled" }}>
                    （{transfers.venue === "市場外" ? "市場外・" : ""}
                    {/* 新株予約権等は1株あたりの株価ではないため、種類を明示して単位も変える */}
                    {transfers.isEquity ? "1株" : `${transfers.securityType ?? "1単位"} 1単位`}
                    {transfers.unitPrice.toLocaleString("ja-JP")}円
                    {transfers.shares > 0 &&
                      ` × ${transfers.shares.toLocaleString("ja-JP")}${transfers.isEquity ? "株" : ""}`}
                    ）
                  </Typography>
                )}
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
        {/* 記事は入口160セッション（TOPに次ぐ2位）なのに滞在16秒で、本文下の回遊導線までは
            到達していない（2026-08-27のGA4実測）。要点を読んだ直後に次のページを出す。 */}
        <ArticleNextStep
          stockName={article.stockName}
          stockHref={stockPageHref}
          filerName={filerName ?? undefined}
          filerHref={filerPageHref}
          dealDate={article.dealDate}
          dateHref={datePageHref}
        />
        {/* 全記事で同一の説明文（提出期限のズレ・免責・分類の定義）はここに書かず、
            FAQと/aboutへのリンクに寄せる。同じ定型文が全記事の本文比率を押し上げると
            「他ページと内容が酷似」と判定されるため（GSCのクロール済み-未登録の主因）。 */}
        <p className="-mt-2 mb-6 text-xs text-ink-muted">
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
        {filerName && (
          <HoldingRatioChart
            filerName={displayFilerName(filerName)}
            stockName={article.stockName}
            points={holdingHistory}
          />
        )}
        {/* AEO（回答エンジン最適化）用の質問型見出し＋直答。強調スニペット／AI検索は
            「質問見出しの直下にある40〜60字の1文」を抜き出すため、構造化データから
            組み立てた事実だけの1文を本文の前に置く。取れない項目は省いて文を短くする。 */}
        {filerName && holdingRatio !== null && (
          <section className="mb-6">
            <h2 className="mb-2 text-xl font-bold text-brand-navy">この開示で何が起きた？</h2>
            <p className="m-0 text-base leading-relaxed text-ink-secondary">
              {formatDate(article.dealDate)}、{displayFilerName(filerName)}が{article.stockName}（{article.stockCode}）の
              保有比率を
              {snapshot?.holdingRatioPrior != null
                ? `${formatRatio(snapshot.holdingRatioPrior)}%から${formatRatio(holdingRatio)}%へ`
                : `${formatRatio(holdingRatio)}%と`}
              {ratioChange === null || ratioChange === 0
                ? "報告しました"
                : ratioChange > 0
                  ? "引き上げました"
                  : "引き下げました"}
              （EDINET大量保有報告書）。
              {transfers && transfers.counterparties.length > 0 && (
                <>
                  譲渡の相手方は{transfers.counterparties.slice(0, 3).join("、")}
                  {transfers.unitPrice !== null &&
                    `（${transfers.isEquity ? "1株" : `${transfers.securityType ?? "1単位"} 1単位`}${transfers.unitPrice.toLocaleString("ja-JP")}円）`}
                  です。
                </>
              )}
            </p>
          </section>
        )}
        <div
          className="prose max-w-none prose-headings:text-brand-navy prose-a:text-brand-blue first:prose-p:first-letter:float-left first:prose-p:first-letter:mr-2 first:prose-p:first-letter:text-5xl first:prose-p:first-letter:font-bold first:prose-p:first-letter:text-brand-navy"
          dangerouslySetInnerHTML={{ __html: linkedBody }}
        />
        <div className="mt-8 rounded-md border border-rule bg-section-tint px-4 py-3 text-xs leading-relaxed text-ink-tertiary">
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
          <p className="m-0 mt-1">
            {exactAmountOku !== null
              ? "金額は開示された譲渡単価×株数です。"
              : "金額は発行済株式数と株価からの概算です。"}
          </p>
        </div>
        {tags && tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-x-3 gap-y-1 border-t border-rule pt-4 text-xs text-ink-tertiary">
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
            {stockPageHref && (
              <div className="mt-3">
                <ActionButton href={stockPageHref}>
                  この銘柄の大量保有・自社株買い履歴をすべて見る
                </ActionButton>
              </div>
            )}
          </div>
        )}
        {filerName && (
          <div className="mt-10 border-t border-rule pt-6">
            <h2 className="mb-2 text-xl font-bold text-brand-navy">
              {displayFilerName(filerName)}とはどんな投資家？
            </h2>
            {article.dealType && (
              <p className="mb-3 text-sm leading-relaxed text-ink-secondary">
                {displayFilerName(filerName)}は「{category}」に分類される投資家です。
                {DEAL_TYPE_DESCRIPTIONS[article.dealType]}
              </p>
            )}
            {filerPageHref && (
              <ActionButton href={filerPageHref}>
                {displayFilerName(filerName)}の保有銘柄・比率推移を見る
              </ActionButton>
            )}
          </div>
        )}
        <div className="mt-10 border-t border-rule pt-6">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">関連ランキング</h2>
          <nav aria-label="関連ランキング" className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
            <Link href="/ranking/returns" className="text-brand-blue hover:underline">3ヶ月リターンランキング</Link>
            <Link href="/trending" className="text-brand-blue hover:underline">銘柄ランキング</Link>
            {datePageHref && (
              <Link href={datePageHref} className="text-brand-blue hover:underline">
                {formatDate(article.dealDate)}の全開示
              </Link>
            )}
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

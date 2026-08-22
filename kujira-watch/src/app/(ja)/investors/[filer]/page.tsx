import type { Metadata } from "next";
import Link from "next/link";
import { notFound, permanentRedirect } from "next/navigation";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import DealTypeBadge from "@/components/DealTypeBadge";
import RatioTransition from "@/components/RatioTransition";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { disclosureDocLabel, edinetPdfUrl } from "@/lib/disclosures";
import {
  getAllFilers,
  getFilerClassification,
  getFilerHoldings,
  getFilerIdByName,
  getFilerNameById,
  investorPath,
} from "@/lib/investors";
import { displayFilerName, formatDate } from "@/lib/format";
import { SITE_URL } from "@/lib/site";
import AdUnit from "@/components/AdUnit";
import FilerReturnRecord from "@/components/FilerReturnRecord";
import FollowUpdatesCta from "@/components/FollowUpdatesCta";
import FaqAccordionList from "@/components/FaqAccordionList";
import { buildInvestorFaqItems } from "@/lib/investorFaq";
import { getFilerReturnPositions, formatSignedPercent } from "@/lib/investorReturns";

// データ源はEDINETの日次開示。5分周期の再検証はクローラーのアクセスがほぼ毎回
// 再生成に当たりコールドTTFBが1.4〜3.5秒になっていた（クロール速度の律速）ため1日にする。
export const revalidate = 86400;

// generateStaticParams が無い動的セグメントはNext 16ではリクエスト毎のSSRになり、
// 何度アクセスしてもCDNキャッシュに乗らない（実測: x-vercel-cache: MISS・no-store）。
// 一部でも事前生成しておくとルート全体がISR扱いになり、事前生成していないパラメータも
// 2回目以降はCDNから返る。クロール速度に直結するので主要分だけ事前生成する。
const PRERENDERED_FILERS = 300;

// URLは /investors/<連番ID>（edinet_filer_ids）。以前は /investors/<提出者名> で、日本語・
// 全角・空白を含む長いURLはGoogleに登録されず（2026-08-23のSearch Consoleカバレッジで
// 投資家ページ603件が未登録）、事前生成時もURLエンコード後のファイル名が255バイト制限に
// 掛かってビルドが落ちていた（ENAMETOOLONG、2026-08-18）。番号化で両方とも解消する。
// 旧URL（提出者名）で来たアクセスは下のresolveFilerで番号URLへ301転送する。
export async function generateStaticParams() {
  // 一覧の取得に失敗しても空配列を返してビルドは通す（microCMS/Supabaseの一時障害で
  // デプロイ全体を落とさないため）。空でもルートはISR扱いのままになる。
  try {
    const filers = await getAllFilers();
    return filers
      .filter((filer) => filer.filerId !== null)
      .sort((a, b) => b.holdingCount - a.holdingCount)
      .slice(0, PRERENDERED_FILERS)
      .map((filer) => ({ filer: String(filer.filerId) }));
  } catch {
    return [];
  }
}

// URLパラメータから提出者を解決する。番号なら edinet_filer_ids を引き、旧形式（提出者名）
// なら名前→番号を引いて正規URLへ301転送する。どちらも該当が無ければnull（=404）。
async function resolveFiler(filer: string): Promise<{ filerId: number; filerName: string } | null> {
  if (/^\d+$/.test(filer)) {
    const filerId = Number(filer);
    const filerName = await getFilerNameById(filerId);
    return filerName ? { filerId, filerName } : null;
  }
  const filerName = decodeURIComponent(filer);
  const filerId = await getFilerIdByName(filerName);
  if (filerId === null) return null;
  permanentRedirect(investorPath(filerId, filerName));
}

type Props = {
  params: Promise<{ filer: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { filer } = await params;
  // 旧URLの転送はページ本体(resolveFiler)に任せ、metadata側は空を返す。
  if (!/^\d+$/.test(filer)) return {};
  const filerName = await getFilerNameById(Number(filer));
  if (!filerName) return {};
  const holdings = await getFilerHoldings(filerName);
  if (holdings.length === 0) return {};

  const title = `${filerName}の大量保有報告書・保有銘柄一覧`;
  const description = `${filerName}がEDINET大量保有報告書（5%ルール）で開示した保有銘柄・保有比率の推移を${holdings.length}件まとめました。`;
  const url = `${SITE_URL}/investors/${filer}`;

  return {
    title,
    description,
    alternates: {
      canonical: url,
      types: { "application/rss+xml": `${url}/feed.xml` },
    },
    openGraph: { title, description, url },
  };
}

export default async function InvestorPage({ params }: Props) {
  const { filer } = await params;
  const resolved = await resolveFiler(filer);
  if (!resolved) notFound();
  const { filerName } = resolved;

  const [holdings, classification, returnPositions] = await Promise.all([
    getFilerHoldings(filerName),
    getFilerClassification(filerName),
    // 開示テーブルの「3ヶ月後」列。買い開示のうち3ヶ月が経過したものだけ値が入る。
    getFilerReturnPositions(filerName),
  ]);

  if (holdings.length === 0) {
    notFound();
  }

  const url = `${SITE_URL}/investors/${filer}`;
  const category = classification?.category ?? "その他";

  // holdingsはdisc_date降順のため、issuerCodeごとに最初に出現したもの(=最新の開示)を採用する。
  // 表示順は保有比率の高い順（比率不明は末尾）。FAQや買い増し/売却リストも同じ順に揃う。
  const majorHoldings = Array.from(
    new Map(holdings.map((h) => [h.issuerCode, h])).values()
  ).sort((a, b) => (b.holdingRatio ?? -1) - (a.holdingRatio ?? -1));
  // 最新開示の増減方向で「買い増し・新規」「売却」に分ける（前回比率が取れない開示は方向不明として除外）。
  const direction = (h: (typeof majorHoldings)[number]) =>
    h.holdingRatio !== null && h.holdingRatioPrior !== null
      ? h.holdingRatio - h.holdingRatioPrior
      : null;
  const recentBuys = majorHoldings.filter((h) => {
    const d = direction(h);
    return d !== null && d > 0;
  });
  const recentSells = majorHoldings.filter((h) => {
    const d = direction(h);
    return d !== null && d < 0;
  });

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: "投資家一覧", item: `${SITE_URL}/investors` },
      { "@type": "ListItem", position: 3, name: filerName, item: url },
    ],
  };

  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${filerName}の保有銘柄一覧`,
    itemListElement: holdings.map((h, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: `${h.issuerName}（${h.issuerCode}）`,
      url: `${SITE_URL}/stocks/${h.issuerCode}`,
    })),
  };

  const faqItems = buildInvestorFaqItems(filerName, majorHoldings, recentBuys);
  const faqJsonLd =
    faqItems.length > 0
      ? {
          "@context": "https://schema.org",
          "@type": "FAQPage",
          mainEntity: faqItems.map((faq) => ({
            "@type": "Question",
            name: faq.question,
            acceptedAnswer: { "@type": "Answer", text: faq.answer },
          })),
        }
      : null;

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
      {faqJsonLd && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
        />
      )}
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href="/investors" className="hover:text-brand-blue">投資家一覧</Link>
        {" / "}
        <span className="text-foreground/70">{displayFilerName(filerName)}</span>
      </nav>
      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{displayFilerName(filerName)}</h1>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <DealTypeBadge dealType={category} />
        {classification?.description && (
          <p className="text-sm text-foreground/60">{classification.description}</p>
        )}
      </div>
      {classification?.profile && (
        <div className="mb-8 border-t border-rule pt-4">
          <h2 className="mb-2 text-sm font-bold text-brand-navy">{displayFilerName(filerName)}について</h2>
          <p className="whitespace-pre-line text-sm leading-relaxed text-foreground/70">
            {classification.profile}
          </p>
        </div>
      )}
      {/* 「この投資家についていって大丈夫か」に最初に答える数字なので、プロフィールの直後・
          保有銘柄の前に置く。/ranking/returnsと同じ行を読むので数字は必ず一致する。 */}
      <FilerReturnRecord filerName={filerName} />
      {majorHoldings.length > 0 && (
        <div className="mb-8 border-t border-rule pt-4">
          <h2 className="mb-2 text-sm font-bold text-brand-navy">主な保有銘柄</h2>
          <ul className="space-y-2 text-sm">
            {majorHoldings.map((h) => (
              <li key={h.issuerCode}>
                <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                  {h.issuerName}（{h.issuerCode}）
                </Link>
                {h.holdingRatio !== null && (
                  <span className="ml-2 text-xs text-foreground/40">保有比率{h.holdingRatio}%</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {(recentBuys.length > 0 || recentSells.length > 0) && (
        <div className="mb-8 grid grid-cols-1 gap-6 border-t border-rule pt-4 sm:grid-cols-2">
          {recentBuys.length > 0 && (
            <div>
              <h2 className="mb-2 text-sm font-bold text-brand-navy">直近で買い増した銘柄</h2>
              <ul className="space-y-2 text-sm">
                {recentBuys.map((h) => (
                  <li key={h.issuerCode}>
                    <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                      {h.issuerName}（{h.issuerCode}）
                    </Link>
                    <span className="ml-2 text-xs">
                      <RatioTransition ratio={h.holdingRatio} prior={h.holdingRatioPrior} />
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {recentSells.length > 0 && (
            <div>
              <h2 className="mb-2 text-sm font-bold text-brand-navy">直近で売却した銘柄</h2>
              <ul className="space-y-2 text-sm">
                {recentSells.map((h) => (
                  <li key={h.issuerCode}>
                    <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                      {h.issuerName}（{h.issuerCode}）
                    </Link>
                    <span className="ml-2 text-xs">
                      <RatioTransition ratio={h.holdingRatio} prior={h.holdingRatioPrior} />
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      <div className="mb-6 border-t border-rule pt-4">
        <h2 className="text-xl font-bold text-brand-navy">最近の取引</h2>
        <p className="mt-1 text-sm text-foreground/50">
          EDINET大量保有報告書（5%ルール）にもとづき、{filerName}が開示した保有銘柄・保有比率の推移を
          {holdings.length}件まとめています。「3ヶ月後」は買い増し・新規取得の開示について、開示日の終値から
          63営業日後の終値までの騰落率です（売却の開示・訂正報告書・まだ3ヶ月経っていない開示は「—」）。
          個別銘柄の詳しい解説記事は各銘柄ページからご覧いただけます。
        </p>
      </div>
      <TableContainer sx={{ borderTop: 1, borderColor: "divider" }}>
        <Table size="small" sx={{ minWidth: 640, "& .MuiTableCell-root": { borderColor: "divider" } }}>
          <TableHead>
            <TableRow>
              <TableCell sx={{ color: "text.disabled" }}>開示日</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>銘柄</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>種別</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>保有比率</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>3ヶ月後</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>原文</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {holdings.map((h) => (
              <TableRow key={h.docId}>
                <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                  {formatDate(h.discDate)}
                </TableCell>
                <TableCell>
                  <Link href={`/stocks/${h.issuerCode}`} className="text-brand-blue hover:underline">
                    {h.issuerName}（{h.issuerCode}）
                  </Link>
                </TableCell>
                <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                  {disclosureDocLabel(h)}
                </TableCell>
                <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                  <RatioTransition ratio={h.holdingRatio} prior={h.holdingRatioPrior} />
                </TableCell>
                {/* 開示日の終値から63営業日後の終値までの騰落率。売却の開示・訂正報告書・
                    まだ3ヶ月経っていない開示は明細ビューに無いので空欄（—）になる。
                    突合はdocIdなので、同日に変更報告書と訂正報告書が並んでも
                    実際に集計された1行にしか数字が出ない。 */}
                <TableCell sx={{ whiteSpace: "nowrap" }}>
                  {(() => {
                    const ret = returnPositions[h.docId];
                    if (ret === undefined) return <span className="text-foreground/30">—</span>;
                    return (
                      <span
                        className={`text-xs font-bold tabular-nums ${ret >= 0 ? "text-gain" : "text-loss"}`}
                      >
                        {formatSignedPercent(ret)}
                      </span>
                    );
                  })()}
                </TableCell>
                {/* 一次ソース（EDINETの原文PDF）への直リンク。以前は/disclosuresだけが
                    持っていたが、同ページ廃止にあたり銘柄・投資家ページへ移した。 */}
                <TableCell sx={{ whiteSpace: "nowrap" }}>
                  <a
                    href={edinetPdfUrl(h.docId)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-brand-blue hover:underline"
                  >
                    PDF ↗
                  </a>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      {category !== "その他" && (
        <p className="mt-6 text-xs text-foreground/40">
          {DEAL_TYPE_DESCRIPTIONS[category]}
        </p>
      )}
      <div className="mt-8">
        <FollowUpdatesCta
          feedUrl={`/investors/${filer}/feed.xml`}
          targetLabel={displayFilerName(filerName)}
        />
      </div>
      {faqItems.length > 0 && (
        <div className="mt-8 border-t border-rule pt-4">
          <h2 className="mb-2 text-xl font-bold text-brand-navy">よくある質問</h2>
          <FaqAccordionList faqs={faqItems} />
        </div>
      )}
      <AdUnit placement="bottom" />
    </div>
  );
}

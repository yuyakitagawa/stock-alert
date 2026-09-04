import type { DealType } from "@/types/article";
import { isCorrectionArticle } from "@/lib/format";

// 英語版（en.kujira-watch.com）専用の定数・整形関数。
//
// 英語版は2026-08-29に /en 配下を廃止したが、microCMSに残っている英訳済み記事
// （titleEn/bodyEn。廃止時点で約1,000本）をサブドメインで配信し直し、
// 「英語面にクローラーが来るか」を blog_crawler_log の host 列で実測するために復活させた。
// 新規記事の英訳は生成していない（Anthropic APIの消化を増やさないため）ので、
// 英語版に載るのは廃止前に英訳された記事だけ。
//
// 日本語側のコンポーネントに locale 引数を戻すと変更範囲が全面に広がるため、
// 英語版は src/app/(en)/en 配下で完結させ、共通で要るものだけをこのファイルに置く。
// proxy.ts（Edge）からも import するので、サーバー専用モジュールを import しないこと。

// 英語版のホスト名。Vercelのドメイン設定とDNS（CNAME en → cname.vercel-dns.com）は手動。
export const EN_HOST = process.env.NEXT_PUBLIC_EN_HOST || "en.kujira-watch.com";
export const EN_SITE_URL = `https://${EN_HOST}`;

// 英語ブランド名。ドメイン(kujira-watch.com)とは別物として管理する（日本語側と同じ扱い）。
export const SITE_NAME_EN = "Big Investor Watch";

export const SITE_DESCRIPTION_EN =
  'A blog tracking Japanese-market "whales" — institutional investors, activists, insiders, and buybacks — based on EDINET large-shareholding filings (Japan\'s 5% rule).';

// 投資家分類の英語ラベルと説明文。dealTypeの値（日本語、edinet_filer_classificationマスターと
// 1対1）自体は変更しない — 表示用の対訳レイヤー。
export const DEAL_TYPE_EN: Record<DealType, { label: string; description: string }> = {
  個人: {
    label: "Individual",
    description: "Holdings under an individual's name, such as an executive or major shareholder.",
  },
  創業家の資産管理会社: {
    label: "Founder-Family Holding Company",
    description: "A privately held asset-management company owned by a company's founding family.",
  },
  "公益/一般財団法人": {
    label: "Public-Interest / General Foundation",
    description: "Non-profit public-interest or general incorporated foundations.",
  },
  プライムブローカー: {
    label: "Prime Broker",
    description:
      "Held via a securities firm's prime brokerage business (often client short-sale collateral, not necessarily an investment decision).",
  },
  アクティビスト: {
    label: "Activist Investor",
    description: "A fund seeking management influence or proposing shareholder resolutions.",
  },
  VC: {
    label: "Venture Capital",
    description: "Venture capital — often a pre-IPO stake retained after listing.",
  },
  "PE・メザニンファンド": {
    label: "PE / Mezzanine Fund",
    description: "Private equity, buyout, or mezzanine funds.",
  },
  独立系ブティックAM: {
    label: "Independent Boutique Asset Manager",
    description: "A domestic independent asset manager not affiliated with a megabank or insurer group.",
  },
  国内アセットマネジメント: {
    label: "Domestic Asset Manager",
    description:
      "A traditional asset manager affiliated with a megabank or insurer group (including trust-bank trust accounts).",
  },
  外資系伝統運用会社: {
    label: "Foreign Traditional Asset Manager",
    description: "A large, diversified asset manager headquartered overseas.",
  },
  日系証券銀行: {
    label: "Japanese Securities Firm / Bank",
    description: "A Japanese securities firm or bank itself (not its asset-management subsidiary).",
  },
  事業会社: {
    label: "Operating Company",
    description: "A general (non-financial) operating company, domestic or foreign.",
  },
  その他: {
    label: "Other",
    description: "A filer that does not clearly fit the categories above, or could not be determined.",
  },
  自社株買い: {
    label: "Share Buyback",
    description:
      "The issuer's own share repurchase (TDnet timely disclosure), showing the board-approved upper limit.",
  },
};

export function dealTypeLabelEn(dealType: DealType | undefined): string | undefined {
  // microCMSのdealTypeはセレクト型で、選択肢に無い値は空で保存される。undefinedでも落とさない
  // （実害: 2026-08-24〜25、旧/enのビルドが undefined.label で24時間止まった）。
  return dealType ? DEAL_TYPE_EN[dealType]?.label : undefined;
}

export function formatDateEn(dateString: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    timeZone: "Asia/Tokyo",
  }).format(new Date(dateString));
}

// amount は億円(100,000,000円)単位。英語版は短縮表記(¥X.XB / ¥XXXM)にする。
export function formatDealAmountEn(amount: number): string {
  const millionYen = amount * 100;
  return millionYen >= 1000 ? `¥${(millionYen / 1000).toFixed(1)}B` : `¥${millionYen.toFixed(0)}M`;
}

// 訂正報告書の記事は金額を持たない（dealAmount=0）。日本語側の formatDealAmountOrCorrection と同じ扱い。
export function formatDealAmountOrCorrectionEn(article: { dealAmount: number; tags?: string }): string {
  if (isCorrectionArticle(article.tags)) return "Correction";
  return formatDealAmountEn(article.dealAmount);
}

export function isTranslated(article: { titleEn?: string; bodyEn?: string }): boolean {
  return Boolean(article.titleEn && article.bodyEn);
}

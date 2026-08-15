import type { DealType } from "@/types/article";

// 投資家分類ごとの表示色（ドット/文字）。DealTypeBadge（Chip+Tooltip、"use client"）と
// DealTypeLabel（一覧向けの軽量なサーバーコンポーネント）で同じ色を使うため、
// コンポーネント側ではなくここに置く。
export const DEAL_TYPE_COLORS: Record<DealType, { dot: string; text: string }> = {
  個人: { dot: "#d97706", text: "#92400e" },
  創業家の資産管理会社: { dot: "#f97316", text: "#c2410c" },
  "公益/一般財団法人": { dot: "#0d9488", text: "#0f766e" },
  プライムブローカー: { dot: "#6366f1", text: "#4338ca" },
  アクティビスト: { dot: "#e11d48", text: "#be123c" },
  VC: { dot: "#ec4899", text: "#db2777" },
  "PE・メザニンファンド": { dot: "#c026d3", text: "#a21caf" },
  独立系ブティックAM: { dot: "#3b82f6", text: "#2563eb" },
  国内アセットマネジメント: { dot: "#2563eb", text: "#1d4ed8" },
  外資系伝統運用会社: { dot: "#7c3aed", text: "#6d28d9" },
  日系証券銀行: { dot: "#0284c7", text: "#0369a1" },
  事業会社: { dot: "#059669", text: "#047857" },
  その他: { dot: "#9ca3af", text: "rgba(32, 29, 26, 0.6)" },
};

// web/publish_blog_articles.py の classify_filer() が使う判断基準と表現をそろえた説明文。
// 分類の定義そのものを変えるものではなく、初心者向けに意味を補足するための表示用データ。
export const DEAL_TYPE_DESCRIPTIONS: Record<DealType, string> = {
  個人: "役員・大株主など個人名義での保有。",
  創業家の資産管理会社: "創業家・オーナー一族が保有する非上場の資産管理会社。",
  "公益/一般財団法人": "公益法人・一般財団法人など非営利法人。",
  プライムブローカー: "証券会社のプライムブローカー業務での保有（顧客の空売り担保等が主で、必ずしも投資判断ではない）。",
  アクティビスト: "経営への関与や株主提案を目的とするファンド。",
  VC: "ベンチャーキャピタル。未上場段階からの出資が上場後も残っているケースが多い。",
  "PE・メザニンファンド": "プライベートエクイティ・バイアウト・メザニンファンド。",
  独立系ブティックAM: "メガバンク・保険系列に属さない国内の独立系運用会社。",
  国内アセットマネジメント: "メガバンク・保険系列の伝統的な資産運用会社（信託銀行の信託口を含む）。",
  外資系伝統運用会社: "海外に拠点を置く大手の分散型資産運用会社。",
  日系証券銀行: "日本の証券会社・銀行本体（資産運用子会社ではなく本体としての保有）。",
  事業会社: "金融業ではない一般の事業会社（国内外問わず）。",
  その他: "上記のいずれにも明確に当てはまらない、または判定できなかった提出者。",
};

// 英語版（/en）用の投資家分類ラベル・URLスラッグ・説明文。
// dealTypeの値（日本語、edinet_filer_classificationマスターと1対1）自体は変更しない —
// あくまで表示・URL用の対訳レイヤー。
export const DEAL_TYPE_EN: Record<DealType, { label: string; slug: string; description: string }> = {
  個人: {
    label: "Individual",
    slug: "individual",
    description: "Holdings under an individual's name, such as an executive or major shareholder.",
  },
  創業家の資産管理会社: {
    label: "Founder-Family Holding Company",
    slug: "founder-family-holding",
    description: "A privately held asset-management company owned by a company's founding family.",
  },
  "公益/一般財団法人": {
    label: "Public-Interest / General Foundation",
    slug: "public-interest-foundation",
    description: "Non-profit public-interest or general incorporated foundations.",
  },
  プライムブローカー: {
    label: "Prime Broker",
    slug: "prime-broker",
    description: "Held via a securities firm's prime brokerage business (often client short-sale collateral, not necessarily an investment decision).",
  },
  アクティビスト: {
    label: "Activist Investor",
    slug: "activist-investor",
    description: "A fund seeking management influence or proposing shareholder resolutions.",
  },
  VC: {
    label: "Venture Capital",
    slug: "venture-capital",
    description: "Venture capital — often a pre-IPO stake retained after listing.",
  },
  "PE・メザニンファンド": {
    label: "PE / Mezzanine Fund",
    slug: "pe-mezzanine-fund",
    description: "Private equity, buyout, or mezzanine funds.",
  },
  独立系ブティックAM: {
    label: "Independent Boutique Asset Manager",
    slug: "independent-boutique-am",
    description: "A domestic independent asset manager not affiliated with a megabank or insurer group.",
  },
  国内アセットマネジメント: {
    label: "Domestic Asset Manager",
    slug: "domestic-asset-manager",
    description: "A traditional asset manager affiliated with a megabank or insurer group (including trust-bank trust accounts).",
  },
  外資系伝統運用会社: {
    label: "Foreign Traditional Asset Manager",
    slug: "foreign-traditional-asset-manager",
    description: "A large, diversified asset manager headquartered overseas.",
  },
  日系証券銀行: {
    label: "Japanese Securities Firm / Bank",
    slug: "japanese-securities-bank",
    description: "A Japanese securities firm or bank itself (not its asset-management subsidiary).",
  },
  事業会社: {
    label: "Operating Company",
    slug: "operating-company",
    description: "A general (non-financial) operating company, domestic or foreign.",
  },
  その他: {
    label: "Other",
    slug: "other",
    description: "A filer that does not clearly fit the categories above, or could not be determined.",
  },
};

export const EN_SLUG_TO_DEAL_TYPE: Record<string, DealType> = Object.fromEntries(
  (Object.entries(DEAL_TYPE_EN) as [DealType, { label: string; slug: string; description: string }][]).map(
    ([dealType, info]) => [info.slug, dealType]
  )
);

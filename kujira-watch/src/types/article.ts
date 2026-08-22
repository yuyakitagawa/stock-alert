import type { MicroCMSImage, MicroCMSListContent, MicroCMSObjectContent } from "microcms-js-sdk";

// EDINET提出者(投資家)分類。web/publish_blog_articles.pyのclassify_filer()が参照する
// Supabase edinet_filer_classificationマスターと1対1で対応する（値そのものがカテゴリ名で
// あり「買い」等の接尾辞は付けない）。
export type DealType =
  | "個人"
  | "創業家の資産管理会社"
  | "公益/一般財団法人"
  | "プライムブローカー"
  | "アクティビスト"
  | "VC"
  | "PE・メザニンファンド"
  | "独立系ブティックAM"
  | "国内アセットマネジメント"
  | "外資系伝統運用会社"
  | "日系証券銀行"
  | "事業会社"
  | "その他"
  // 提出者分類ではなく、発行体自身による自社株買い（TDnet適時開示。web/publish_buyback_articles.py）。
  // 「誰が買ったか」の軸で大量保有報告書と同じ一覧・フィルターに並べるため、dealTypeの値として扱う。
  | "自社株買い";

// 機関投資家系（プライムブローカー〜日系証券銀行）をサイト上部のカテゴリフィルターの
// 左端に置きたいという要望から、この並び順にしている。
export const DEAL_TYPES: DealType[] = [
  "プライムブローカー",
  "アクティビスト",
  "VC",
  "PE・メザニンファンド",
  "独立系ブティックAM",
  "国内アセットマネジメント",
  "外資系伝統運用会社",
  "日系証券銀行",
  "個人",
  "創業家の資産管理会社",
  "公益/一般財団法人",
  "事業会社",
  "その他",
  "自社株買い",
];

// サイト上部のカテゴリフィルターはmicroCMS側に別フィールドを持たず、dealTypeの値を
// そのままカテゴリ名として使う（web/publish_blog_articles.pyのclassify_filer()分類と
// 1対1で対応するため、CMS側の選択肢を別途同期させる必要が無い）。
export function categoryLabel(dealType: DealType): string {
  return dealType;
}

export const CATEGORIES: string[] = DEAL_TYPES;

export const DEAL_TYPE_BY_CATEGORY: Record<string, DealType> = Object.fromEntries(
  DEAL_TYPES.map((dealType) => [dealType, dealType])
);

export type Article = {
  title: string;
  body: string;
  // 英語版（/en）用の翻訳フィールド。未設定の記事はEN側の一覧・詳細には出さない
  // （日英混在ページを検索エンジンに出さないため）。
  titleEn?: string;
  bodyEn?: string;
  stockName: string;
  stockCode: string;
  dealType: DealType;
  dealDate: string;
  dealAmount: number;
  sourceUrl?: string;
  tags?: string;
  eyecatch?: MicroCMSImage;
  // 取引を行った投資家（提出者）名。過去記事には設定されていない場合がある。
  filerName?: string;
  // 保有比率の変化幅（ポイント）。買いは正・売りは負。2026-08-15にスキーマ追加した
  // フィールドのため、それ以前の記事には無い。記事詳細のファクトボックス「前回比」に使う。
  ratioChangePct?: number;
};

export type ArticleContent = Article & MicroCMSListContent;

// /about ページ用（microCMSオブジェクト形式API）。methodology/faqは未入力の場合がある任意項目。
export type AboutPage = {
  heroTitle: string;
  heroLead: string;
  profileBody: string;
  dataSources: string;
  methodology?: string;
  disclaimer: string;
  faq?: string;
};

export type AboutPageContent = AboutPage & MicroCMSObjectContent;

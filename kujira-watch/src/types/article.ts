import type { MicroCMSImage, MicroCMSListContent } from "microcms-js-sdk";

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
  | "その他";

export const DEAL_TYPES: DealType[] = [
  "個人",
  "創業家の資産管理会社",
  "公益/一般財団法人",
  "プライムブローカー",
  "アクティビスト",
  "VC",
  "PE・メザニンファンド",
  "独立系ブティックAM",
  "国内アセットマネジメント",
  "外資系伝統運用会社",
  "日系証券銀行",
  "事業会社",
  "その他",
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
  stockName: string;
  stockCode: string;
  dealType: DealType;
  dealDate: string;
  dealAmount: number;
  sourceUrl?: string;
  tags?: string;
  eyecatch?: MicroCMSImage;
};

export type ArticleContent = Article & MicroCMSListContent;

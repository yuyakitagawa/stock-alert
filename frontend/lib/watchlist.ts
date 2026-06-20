// 値上げ力ウォッチリスト（toC独占 × インフレ耐性）のメタ情報。
// 独占商品・独占率・海外比率はキュレーション（公知ベースの概算）。
// ウォッチリストでブックマーク銘柄に該当があれば表示する（無ければ「—」）。
// 正本: data/pricing_power_watchlist.csv

export interface WatchMeta {
  category:      string;
  product:       string;
  domesticShare: string;   // 定性表現（"国内首位 約50%" 等）
  overseasRatio: number;   // 海外売上比率（概算 %）
  note:          string;
}

export const PRICING_POWER_META: Record<string, WatchMeta> = {
  "2811": { category: "食品",       product: "ケチャップ・野菜ジュース", domesticShare: "国内首位 約50%",  overseasRatio: 40, note: "トマト加工で圧倒的ブランド。健康志向で値上げ転嫁が効く" },
  "2229": { category: "菓子",       product: "ポテトチップス",          domesticShare: "国内 約50%",      overseasRatio: 20, note: "スナック寡占。値上げ常習でも数量を維持" },
  "2897": { category: "食品",       product: "カップ麺",               domesticShare: "国内 約40%",      overseasRatio: 35, note: "カップ麺の元祖ブランド。海外伸長で成長余地" },
  "2502": { category: "飲料",       product: "ビール(スーパードライ)",  domesticShare: "国内寡占",        overseasRatio: 50, note: "業界主導の値上げ。欧州買収で海外比率上昇" },
  "8113": { category: "日用品",     product: "紙おむつ・生理用品",       domesticShare: "国内首位",        overseasRatio: 60, note: "アジア新興国で高成長。必需品で景気耐性" },
  "4452": { category: "日用品",     product: "トイレタリー全般",        domesticShare: "国内上位独占級",  overseasRatio: 40, note: "洗剤・紙おむつ等ブランド多数。値上げ＋詰替で防御" },
  "4911": { category: "化粧品",     product: "化粧品",                 domesticShare: "国内首位",        overseasRatio: 60, note: "高価格帯ブランド。インバウンド・中国需要" },
  "4527": { category: "ヘルスケア", product: "目薬・スキンケア",        domesticShare: "国内首位",        overseasRatio: 40, note: "目薬国内トップ。OTC＋スキンケアで価格決定力" },
  "7956": { category: "日用品",     product: "哺乳瓶・ベビー用品",       domesticShare: "国内首位",        overseasRatio: 60, note: "ベビー用品の指名買い。中国・アジア展開" },
  "7309": { category: "機械",       product: "自転車変速機",            domesticShare: "世界 約70-80%",   overseasRatio: 90, note: "自転車部品の世界寡占。実質プライスメーカー" },
};

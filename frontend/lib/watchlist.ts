// 値上げ力ウォッチリスト（toC独占 × インフレ耐性）
// オーナー選定の将来の買い候補。シェア率・海外比率は概算（公知ベース・要検証）。
// 正本: data/pricing_power_watchlist.csv

export interface WatchStock {
  code:           string;
  name:           string;
  category:       string;
  product:        string;
  domesticShare:  string;   // 定性表現（"国内首位 約50%" 等）
  overseasRatio:  number;   // 海外売上比率（概算 %）
  note:           string;
}

export const PRICING_POWER_WATCHLIST: WatchStock[] = [
  { code: "2811", name: "カゴメ",          category: "食品",       product: "ケチャップ・野菜ジュース", domesticShare: "国内首位 約50%",  overseasRatio: 40, note: "トマト加工で圧倒的ブランド。健康志向で値上げ転嫁が効く" },
  { code: "2229", name: "カルビー",        category: "菓子",       product: "ポテトチップス",          domesticShare: "国内 約50%",      overseasRatio: 20, note: "スナック寡占。値上げ常習でも数量を維持" },
  { code: "2897", name: "日清食品HD",      category: "食品",       product: "カップ麺",               domesticShare: "国内 約40%",      overseasRatio: 35, note: "カップ麺の元祖ブランド。海外伸長で成長余地" },
  { code: "2502", name: "アサヒGHD",       category: "飲料",       product: "ビール(スーパードライ)",  domesticShare: "国内寡占",        overseasRatio: 50, note: "業界主導の値上げ。欧州買収で海外比率上昇" },
  { code: "8113", name: "ユニ・チャーム",  category: "日用品",     product: "紙おむつ・生理用品",       domesticShare: "国内首位",        overseasRatio: 60, note: "アジア新興国で高成長。必需品で景気耐性" },
  { code: "4452", name: "花王",            category: "日用品",     product: "トイレタリー全般",        domesticShare: "国内上位独占級",  overseasRatio: 40, note: "洗剤・紙おむつ等ブランド多数。値上げ＋詰替で防御" },
  { code: "4911", name: "資生堂",          category: "化粧品",     product: "化粧品",                 domesticShare: "国内首位",        overseasRatio: 60, note: "高価格帯ブランド。インバウンド・中国需要" },
  { code: "4527", name: "ロート製薬",      category: "ヘルスケア", product: "目薬・スキンケア",        domesticShare: "国内首位",        overseasRatio: 40, note: "目薬国内トップ。OTC＋スキンケアで価格決定力" },
  { code: "7956", name: "ピジョン",        category: "日用品",     product: "哺乳瓶・ベビー用品",       domesticShare: "国内首位",        overseasRatio: 60, note: "ベビー用品の指名買い。中国・アジア展開" },
  { code: "7309", name: "シマノ",          category: "機械",       product: "自転車変速機",            domesticShare: "世界 約70-80%",   overseasRatio: 90, note: "自転車部品の世界寡占。実質プライスメーカー" },
];

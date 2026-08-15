// Google AdSense の設定。パブリッシャーIDが未設定の間は、スクリプト・広告枠・ads.txt の
// すべてが何も出力しない（ローカル開発・プレビュー環境では完全に無効になる）。
//
// NEXT_PUBLIC_ADSENSE_CLIENT: AdSense管理画面の「サイト運営者ID」。`ca-pub-` から始まる形式。
export const ADSENSE_CLIENT = process.env.NEXT_PUBLIC_ADSENSE_CLIENT ?? "";

// 掲載位置。位置ごとに別の広告ユニットを作ることで、AdSense管理画面で収益を分けて見られる。
// スロットIDが未設定の位置には広告を出さないので、片方だけ先に有効化することもできる。
export type AdPlacement = "infeed" | "bottom";

export const ADSENSE_SLOTS: Record<AdPlacement, string> = {
  // 記事一覧(オートスクロール)の取引日グループの区切り
  infeed: process.env.NEXT_PUBLIC_ADSENSE_INFEED_SLOT ?? "",
  // 記事詳細・銘柄ページ等、コンテンツの末尾
  bottom: process.env.NEXT_PUBLIC_ADSENSE_BOTTOM_SLOT ?? "",
};

// ads.txt に書く販売者IDは、スクリプトのclientパラメータと違い `ca-` 接頭辞を付けない。
export const ADSENSE_PUBLISHER_ID = ADSENSE_CLIENT.replace(/^ca-/, "");

// Google AdSense の設定。パブリッシャーIDが未設定の間は、スクリプト・広告枠・ads.txt の
// すべてが何も出力しない（審査申請前・ローカル開発・プレビュー環境では完全に無効になる）。
// 審査に通ってIDが発行されたら、Vercelの環境変数に値を入れるだけで有効化される。
//
// NEXT_PUBLIC_ADSENSE_CLIENT: AdSense管理画面の「サイト運営者ID」。`ca-pub-` から始まる形式。
// NEXT_PUBLIC_ADSENSE_INFEED_SLOT: 記事一覧に差し込む広告ユニットのスロットID（数字列）。
export const ADSENSE_CLIENT = process.env.NEXT_PUBLIC_ADSENSE_CLIENT ?? "";

export const ADSENSE_INFEED_SLOT = process.env.NEXT_PUBLIC_ADSENSE_INFEED_SLOT ?? "";

// ads.txt に書く販売者IDは、スクリプトのclientパラメータと違い `ca-` 接頭辞を付けない。
export const ADSENSE_PUBLISHER_ID = ADSENSE_CLIENT.replace(/^ca-/, "");

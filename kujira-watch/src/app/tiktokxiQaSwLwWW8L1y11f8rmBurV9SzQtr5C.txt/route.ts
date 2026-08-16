// TikTok for Developers のURLプロパティ検証用シグネチャファイル。
// TikTokアプリ（Content Posting API・動画自動投稿）の Terms/Privacy URL 検証で
// kujira-watch.com の所有を証明するために配信する。検証完了後も、TikTok側が
// 定期的に再検証する可能性があるため削除しないこと。
export function GET() {
  return new Response(
    "tiktok-developers-site-verification=xiQaSwLwWW8L1y11f8rmBurV9SzQtr5C\n",
    { headers: { "Content-Type": "text/plain; charset=utf-8" } },
  );
}

import { ADSENSE_CLIENT } from "@/lib/adsense";

// AdSenseのローダー。React 19が `async` + `src` の <script> を <head> へ巻き上げるため、
// bodyに置いてもAdSense側の「サイトにコードを設置したか」の確認・審査は通る。
// 広告枠(InFeedAd)はこのスクリプトが定義する window.adsbygoogle を使う。
export default function AdSenseScript() {
  if (!ADSENSE_CLIENT) return null;

  return (
    <script
      async
      src={`https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT}`}
      crossOrigin="anonymous"
    />
  );
}

import { SITE_NAME } from "@/lib/site";

// 記事詳細の共有ボタン。JSを持ち込まないよう、公式のWeb Intent URLへの
// 素の<a>だけで実装する（SDKを読み込むとページ表示が目に見えて遅くなるため。
// /aboutのX埋め込みタイムラインを撤去したのと同じ理由）。
// 流入をGA4で識別できるようUTMを付ける（自動投稿のutm_source=xとは別値にして、
// 読者による手動シェア経由の流入と区別する）。
export default function ShareButtons({ url, title }: { url: string; title: string }) {
  const shareUrl = `${url}?utm_source=share&utm_medium=social`;
  const encodedUrl = encodeURIComponent(shareUrl);
  const xHref = `https://x.com/intent/post?text=${encodeURIComponent(`${title}｜${SITE_NAME}`)}&url=${encodedUrl}`;
  const lineHref = `https://social-plugins.line.me/lineit/share?url=${encodedUrl}`;
  const hatenaHref = `https://b.hatena.ne.jp/entry/panel/?url=${encodedUrl}`;

  const linkClass =
    "kicker rounded border border-rule px-3 py-1.5 text-brand-navy/70 transition-colors hover:border-brand-blue hover:text-brand-blue";

  return (
    <div className="mt-8 flex flex-wrap items-center gap-2 border-t border-rule pt-5">
      <span className="kicker mr-1 text-foreground/40">この記事を共有</span>
      <a className={linkClass} href={xHref} target="_blank" rel="noopener noreferrer">
        Xでポスト
      </a>
      <a className={linkClass} href={lineHref} target="_blank" rel="noopener noreferrer">
        LINEで送る
      </a>
      <a className={linkClass} href={hatenaHref} target="_blank" rel="noopener noreferrer">
        はてブ
      </a>
    </div>
  );
}

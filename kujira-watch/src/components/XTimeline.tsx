"use client";

import Script from "next/script";

const X_HANDLE = "kujira_watch";

type XTimelineProps = {
  // 指定件数のみ表示（未指定なら公式デフォルトの複数件タイムライン）。
  // 件数固定時はウィジェット側が高さを自動調整するのでdata-heightは付けない。
  tweetLimit?: number;
};

// X公式の埋め込みタイムライン。widgets.jsは<a class="twitter-timeline">を
// 検出してiframeに差し替えるため、リンクはフォールバック表示としてそのまま残す。
export default function XTimeline({ tweetLimit }: XTimelineProps) {
  return (
    <>
      <a
        className="twitter-timeline"
        data-theme="light"
        data-chrome="noheader nofooter noborders transparent"
        data-tweet-limit={tweetLimit}
        {...(tweetLimit ? {} : { "data-height": "600" })}
        href={`https://twitter.com/${X_HANDLE}?ref_src=twsrc%5Etfw`}
      >
        Tweets by {X_HANDLE}
      </a>
      <Script src="https://platform.twitter.com/widgets.js" strategy="lazyOnload" />
    </>
  );
}

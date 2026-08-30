// 一覧ページの<Suspense>フォールバック。
//
// /ranking・/investors・/stocks は `searchParams` を読むため
// リクエストごとの動的レンダリングになり、一覧の描画が終わるまで最初の1バイトも
// 返せていなかった（本番実測でTTFB 0.43〜0.51s。静的ページは0.18〜0.20s）。
// データ取得部分をSuspenseに入れて、見出しまでのシェルを先に流すようにしたときの
// 待ち受け表示がこれ。
//
// MUIのSkeletonを使うとこの骨組みのためだけにクライアントJSが増えるので、素のdivで組む。
export default function ListFallback({ rows = 10 }: { rows?: number }) {
  return (
    <div className="animate-pulse" aria-hidden>
      <div className="mb-4 h-4 w-2/3 rounded-sm bg-foreground/10" />
      <div className="mb-6 flex flex-wrap gap-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-7 w-24 rounded-sm bg-foreground/10" />
        ))}
      </div>
      <div className="border-t border-rule">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="border-b border-rule py-3">
            <div className="h-4 w-1/2 rounded-sm bg-foreground/10" />
            <div className="mt-2 h-3 w-1/3 rounded-sm bg-foreground/10" />
          </div>
        ))}
      </div>
    </div>
  );
}

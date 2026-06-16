"use client";
import Link from "next/link";
import { useMemo } from "react";
import type { Ranking } from "@/lib/types";
import { useBookmarks } from "@/lib/bookmarks";
import { signFmtArrow } from "@/lib/signals";
import BookmarkButton from "@/components/BookmarkButton";

interface Props {
  rankMap: Record<string, Ranking>;
  sectorMap: Record<string, string>;
  asOf: string;
}

function NetBadge({ net }: { net: number }) {
  const up = net >= 0;
  return (
    <span className={`font-mono text-sm font-semibold ${up ? "text-green-400" : "text-red-400"}`}>
      {signFmtArrow(net)}
    </span>
  );
}

function fmt(v: number | null, suffix = "") {
  return v == null ? "—" : `${v.toFixed(1)}${suffix}`;
}

export default function WatchlistClient({ rankMap, sectorMap, asOf }: Props) {
  const { codes, mounted } = useBookmarks();

  const items = useMemo(
    () => codes.map((code) => ({ code, r: rankMap[code], sector: sectorMap[code] })),
    [codes, rankMap, sectorMap],
  );

  // マウント前はSSRと一致させるため何も出さない（localStorage未読込）
  if (!mounted) {
    return <div className="h-40 rounded-xl bg-gray-900/40 border border-gray-800 animate-pulse" />;
  }

  if (codes.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl px-6 py-12 text-center">
        <div className="text-4xl mb-3">🔖</div>
        <p className="text-gray-300 font-medium">ブックマークはまだありません</p>
        <p className="text-sm text-gray-500 mt-2 leading-relaxed">
          気になる銘柄ページや
          <Link href="/rankings" className="text-green-400 hover:underline mx-1">ランキング</Link>
          のしおりアイコン <span className="inline-block align-middle text-gray-400">🔖</span> を押すと、ここに保存されます。
        </p>
        <Link
          href="/rankings"
          className="inline-block mt-5 text-sm font-medium px-4 py-2 rounded-lg bg-green-500/10 border border-green-700 text-green-400 hover:bg-green-500/20 transition-colors"
        >
          ランキングから探す →
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {items.map(({ code, r, sector }) => (
        <Link
          key={code}
          href={`/stocks/${code}`}
          className="block bg-gray-900 border border-gray-800 rounded-xl p-4 hover:bg-gray-800/40 transition-colors"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="font-medium text-white truncate">{r?.name ?? code}</div>
              <div className="text-xs text-gray-600 font-mono">
                {code}
                {sector && <span className="ml-2 text-gray-500">{sector}</span>}
              </div>
            </div>
            <BookmarkButton code={code} />
          </div>

          {r ? (
            <div className="flex items-center justify-between mt-3 gap-3 flex-wrap">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-gray-200">
                  ¥{r.close?.toLocaleString() ?? "—"}
                </span>
                <span className="text-xs font-mono text-gray-500">
                  PER {fmt(r.per, "x")} / PBR {fmt(r.pbr, "x")}
                </span>
              </div>
              <div className="text-right">
                <div className="flex items-center justify-end gap-2">
                  <span className="text-xs text-gray-600">ネット</span>
                  <NetBadge net={r.net} />
                  <span className="text-xs font-mono text-green-400">↑{r.rise_prob.toFixed(0)}%</span>
                  <span className="text-xs font-mono text-red-400">↓{r.drop_prob.toFixed(0)}%</span>
                </div>
                {r.recommend && <div className="text-xs text-gray-500 mt-0.5">{r.recommend}</div>}
              </div>
            </div>
          ) : (
            <div className="mt-3 text-xs text-gray-600">
              当日のAIスコアは未取得（上場廃止・売買停止、または対象外銘柄の可能性）
            </div>
          )}
        </Link>
      ))}

      <p className="text-xs text-gray-600 leading-relaxed pt-1">
        ※ ブックマークはこの端末/ブラウザにのみ保存されます（ログイン不要）。ネットスコア（上昇確率−下落確率）は
        {asOf || "—"} 時点のAIモデル値。しおりアイコンをもう一度押すと削除できます。
      </p>
    </div>
  );
}

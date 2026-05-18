import type { Metadata } from "next";
import { fetchRankings, fetchSparkline } from "@/lib/data";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import StockCard from "@/components/StockCard";
import RecommendBadge from "@/components/RecommendBadge";
import Link from "next/link";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "StockSignal — 日本株AIシグナル",
};

function formatDate(date: string) {
  if (!date) return "—";
  return new Date(date).toLocaleDateString("ja-JP", {
    year: "numeric", month: "long", day: "numeric", weekday: "short",
  });
}

interface SummaryCardProps {
  label: string;
  count: number;
  colorClass: string;
  borderClass: string;
}

function SummaryCard({ label, count, colorClass, borderClass }: SummaryCardProps) {
  return (
    <div className={`bg-gray-900 rounded-xl border ${borderClass} p-5`}>
      <div className={`text-3xl font-bold font-mono ${colorClass}`}>{count}</div>
      <div className="text-sm text-gray-500 mt-1.5">{label}</div>
    </div>
  );
}

const LOGIC_SIGNALS = [
  { signal: "S買い",       desc: "ネット最高値 + 底値実績あり（最強買い）", descEn: "Strongest buy signal" },
  { signal: "A買い",       desc: "ネット高値（買い推奨）",                  descEn: "Buy recommended" },
  { signal: "高値警戒",   desc: "上昇中だが過熱感あり",                    descEn: "Overbought zone" },
  { signal: "方向感なし", desc: "ネット中立（様子見）",                    descEn: "Neutral — wait and see" },
  { signal: "弱気シグナル",desc: "下落傾向あり",                           descEn: "Weak bearish trend" },
  { signal: "下降シグナル",desc: "強い下落圧力",                           descEn: "Strong downtrend" },
  { signal: "売り検討",   desc: "利確・損切りを検討",                     descEn: "Consider profit-taking or stop-loss" },
];

export default async function HomePage() {
  const { date, rows } = await fetchRankings();

  const sBuy    = rows.filter(r => r.recommend === "S買い");
  const aBuy    = rows.filter(r => r.recommend === "A買い");
  const caution = rows.filter(r => r.recommend === "高値警戒");
  const sell    = rows.filter(r => r.recommend === "売り検討");

  const featured = [...sBuy, ...aBuy].slice(0, 12);
  const dateLabel = formatDate(date);

  const sparklines = await Promise.all(featured.map(r => fetchSparkline(r.code)));
  const sparklineMap = Object.fromEntries(featured.map((r, i) => [r.code, sparklines[i]]));

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar dateLabel={dateLabel} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 space-y-10">
        {rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
            <span className="text-5xl">📊</span>
            <p className="text-lg font-medium text-gray-500">本日のデータはまだありません</p>
            <p className="text-sm">平日16時以降に更新されます</p>
          </div>
        ) : (
          <>
            {/* Hero */}
            <section>
              <div className="flex items-baseline gap-3 mb-5">
                <h1 className="text-xl sm:text-2xl font-bold text-white">日本株 シグナル概要</h1>
                <span className="text-sm text-gray-600 font-mono">{dateLabel}</span>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <SummaryCard label="S買い"   count={sBuy.length}    colorClass="text-green-400"  borderClass="border-green-900" />
                <SummaryCard label="A買い"   count={aBuy.length}    colorClass="text-green-500"  borderClass="border-green-900/50" />
                <SummaryCard label="高値警戒" count={caution.length} colorClass="text-yellow-400" borderClass="border-yellow-900/50" />
                <SummaryCard label="売り検討" count={sell.length}    colorClass="text-red-400"    borderClass="border-red-900/50" />
              </div>
            </section>

            {/* Featured stocks */}
            {featured.length > 0 && (
              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-bold text-white">注目銘柄</h2>
                  <Link href="/rankings" className="text-sm text-green-500 hover:text-green-400 transition-colors font-medium">
                    全銘柄を見る →
                  </Link>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                  {featured.map(r => (
                    <StockCard key={r.code} r={r} sparkline={sparklineMap[r.code]} />
                  ))}
                </div>
              </section>
            )}

            {/* Sell signals */}
            {sell.length > 0 && (
              <section>
                <h2 className="text-lg font-bold text-white mb-4">売り検討</h2>
                <div className="bg-red-950/20 border border-red-900/30 rounded-xl divide-y divide-gray-800/60 overflow-hidden">
                  {sell.slice(0, 6).map(r => (
                    <Link
                      key={r.code}
                      href={`/stocks/${r.code}`}
                      className="flex items-center gap-4 px-4 py-3 hover:bg-red-950/30 transition-colors"
                    >
                      <div className="flex-1 min-w-0">
                        <span className="font-semibold text-sm text-white">{r.name}</span>
                        <span className="ml-2 text-xs text-gray-600 font-mono">{r.code}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <RecommendBadge value={r.recommend} />
                        <span className="font-mono text-sm font-bold text-red-400">
                          {(r.net >= 0 ? "+" : "") + r.net.toFixed(1)}%
                        </span>
                      </div>
                    </Link>
                  ))}
                </div>
                {sell.length > 6 && (
                  <p className="text-xs text-gray-600 mt-2 text-right">
                    他 {sell.length - 6} 銘柄 →{" "}
                    <Link href="/rankings?tab=売り検討" className="text-red-500 hover:underline">全て見る</Link>
                  </p>
                )}
              </section>
            )}

            {/* Signal guide */}
            <section className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
              <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide mb-3">シグナル凡例</h2>
              <div className="flex flex-wrap gap-2">
                {["S買い","A買い","高値警戒","方向感なし","弱気シグナル","下降シグナル","売り検討"].map(v => (
                  <RecommendBadge key={v} value={v} />
                ))}
              </div>
            </section>

            {/* Prediction logic */}
            <section className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 space-y-4">
              <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">日本株 予測ロジック</h2>
              <p className="text-sm text-gray-400 leading-relaxed">
                AIモデル（XGBoost）が株価・出来高・相対強度など
                <strong className="text-gray-200"> 34の特徴量</strong>から、
                <strong className="text-gray-200">63日後（約3ヶ月）に±15%以上変動する確率</strong>を予測します。
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-gray-800/60 rounded-lg p-3">
                  <div className="text-green-400 font-semibold mb-0.5">上昇予測 AUC 0.663</div>
                  <div className="text-gray-500">上昇確率を予測</div>
                </div>
                <div className="bg-gray-800/60 rounded-lg p-3">
                  <div className="text-red-400 font-semibold mb-0.5">下落予測 AUC 0.791</div>
                  <div className="text-gray-500">下落確率を予測（高精度）</div>
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 font-semibold mb-2">
                  ネットスコア = 上昇確率 − 下落確率
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-y-2 gap-x-4">
                  {LOGIC_SIGNALS.map(({ signal, desc }) => (
                    <div key={signal} className="flex items-start gap-2 text-xs">
                      <RecommendBadge value={signal} />
                      <span className="text-gray-500 pt-0.5">{desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}

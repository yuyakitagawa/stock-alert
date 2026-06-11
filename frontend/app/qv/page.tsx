import type { Metadata } from "next";
import { fetchRankings, fetchQvSimTrades } from "@/lib/data";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import QvPanel from "@/components/QvPanel";
import QvSimPanel from "@/components/QvSimPanel";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "QVスクリーナー — StockSignal",
  description: "業績強 × 株価低迷（Quality × Value）の逆張り候補銘柄リストと保有管理",
};

// QV条件（バックテストと同一）
const QV_PIOTROSKI_MIN  = 0.67;   // 財務健全（6/9以上）
const QV_POS52_MAX      = 0.45;   // 52週安値圏
const QV_DROP_MAX       = 8.0;    // drop_prob 上限
const QV_VOL_MAX        = 25.0;   // ボラ上限
// 売りシグナル閾値
const QV_SELL_NET_MIN   = -5.0;   // net がこれ未満で売り
const QV_SELL_DROP_MIN  = 10.0;   // drop_prob がこれ以上で売り

export default async function QvPage() {
  const [{ date, rows }, simTrades] = await Promise.all([
    fetchRankings(),
    fetchQvSimTrades(),
  ]);

  // QV条件フィルタリング
  const candidates = rows.filter(r =>
    r.piotroski != null && r.piotroski >= QV_PIOTROSKI_MIN &&
    r.pos52     != null && r.pos52     <  QV_POS52_MAX     &&
    r.drop_prob != null && r.drop_prob <= QV_DROP_MAX      &&
    r.vol       != null && r.vol       <= QV_VOL_MAX       &&
    (r.bps_growth != null && r.bps_growth > 0 || r.eps_surprise != null && r.eps_surprise > 0)
  ).sort((a, b) => b.net - a.net);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 space-y-10">
        <header className="space-y-2">
          <h1 className="text-xl sm:text-2xl font-bold text-white">QV スクリーナー</h1>
          <p className="text-sm text-gray-500">
            業績強（Piotroski ≥ 6/9）× 株価低迷（52週安値圏 45%以内）の逆張り候補。
            週1回チェックして候補があれば検討、90日保有が目安。
          </p>
          <div className="flex flex-wrap gap-4 text-xs text-gray-600 pt-1">
            <span>📅 基準日: <span className="text-gray-400">{date || "—"}</span></span>
            <span>✅ 候補: <span className="text-green-400 font-bold">{candidates.length}</span> 銘柄 / 全 {rows.length.toLocaleString()} 銘柄中</span>
          </div>
        </header>

        <QvPanel
          candidates={candidates}
          allRows={rows}
          date={date}
          sellNetMin={QV_SELL_NET_MIN}
          sellDropMin={QV_SELL_DROP_MIN}
        />

        <hr className="border-gray-800" />

        <QvSimPanel trades={simTrades} />
      </main>
      <Footer />
    </div>
  );
}

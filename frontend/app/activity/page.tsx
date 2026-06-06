import type { Metadata } from "next";
import { fetchActivity } from "@/lib/data";
import type { Activity } from "@/lib/types";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const revalidate = 30;

export const metadata: Metadata = {
  title: "活動ログ — StockSignal",
  description: "AIチーム（FM・Quant・証券アナリスト・Engineer）が今・何をしているか / 何をしたかの活動記録",
};

// ── 担当者・状態の見た目 ──────────────────────────────────────────────────────

const ROLE_STYLE: Record<string, { label: string; cls: string; emoji: string }> = {
  FM:         { label: "ファンドマネージャー", cls: "bg-blue-900/40 text-blue-300 border-blue-800",     emoji: "💼" },
  Quant:      { label: "数量アナリスト",       cls: "bg-purple-900/40 text-purple-300 border-purple-800", emoji: "📐" },
  Securities: { label: "証券アナリスト",       cls: "bg-yellow-900/40 text-yellow-300 border-yellow-800", emoji: "🔍" },
  Engineer:   { label: "エンジニア",           cls: "bg-green-900/40 text-green-300 border-green-800",   emoji: "🔧" },
  System:     { label: "システム",             cls: "bg-gray-800 text-gray-400 border-gray-700",        emoji: "⚙️" },
};

const STATUS_STYLE: Record<string, { label: string; cls: string }> = {
  running:  { label: "実施中",   cls: "bg-amber-500/20 text-amber-300 border-amber-600 animate-pulse" },
  done:     { label: "完了",     cls: "bg-green-600/20 text-green-300 border-green-700" },
  improve:  { label: "改善する", cls: "bg-blue-600/20 text-blue-300 border-blue-700" },
  skip:     { label: "スキップ", cls: "bg-gray-700/40 text-gray-400 border-gray-600" },
  rejected: { label: "不採用",   cls: "bg-orange-600/20 text-orange-300 border-orange-700" },
  failed:   { label: "失敗",     cls: "bg-red-600/20 text-red-300 border-red-700" },
};

function timeLabel(ts: string) {
  return new Date(ts).toLocaleString("ja-JP", {
    month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function ActivityRow({ a }: { a: Activity }) {
  const role   = ROLE_STYLE[a.role]   ?? ROLE_STYLE.System;
  const status = STATUS_STYLE[a.status] ?? { label: a.status, cls: "bg-gray-800 text-gray-400 border-gray-700" };
  const isRunning = a.status === "running";

  return (
    <div className={`relative pl-6 pb-5 border-l ${isRunning ? "border-amber-700" : "border-gray-800"}`}>
      {/* タイムラインの点 */}
      <span
        className={`absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full ${
          isRunning ? "bg-amber-400 animate-pulse" : "bg-gray-700"
        }`}
      />

      <div className="flex items-center gap-2 flex-wrap mb-1.5">
        <span className={`text-xs font-bold px-2 py-0.5 rounded border ${role.cls}`}>
          {role.emoji} {role.label}
        </span>
        <span className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${status.cls}`}>
          {status.label}
        </span>
        <span className="text-[11px] text-gray-600 font-mono ml-auto">{timeLabel(a.ts)}</span>
      </div>

      <p className="text-sm font-medium text-gray-200">{a.step}</p>
      {a.summary && <p className="text-sm text-gray-400 mt-0.5 leading-relaxed">{a.summary}</p>}

      {a.detail && a.detail.trim() && a.detail !== a.summary && (
        <details className="mt-2 group">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300 select-none">
            詳細を見る ▾
          </summary>
          <pre className="mt-2 text-xs text-gray-400 bg-gray-950 border border-gray-800 rounded-lg p-3 whitespace-pre-wrap leading-relaxed font-sans max-h-96 overflow-y-auto">
            {a.detail}
          </pre>
        </details>
      )}
    </div>
  );
}

// ── メインページ ──────────────────────────────────────────────────────────────

export default async function ActivityPage() {
  const items = await fetchActivity(80);

  // 日付ごとにグループ化
  const groups = new Map<string, Activity[]>();
  for (const a of items) {
    const day = a.run_date || a.ts.slice(0, 10);
    if (!groups.has(day)) groups.set(day, []);
    groups.get(day)!.push(a);
  }

  const running = items.filter(a => a.status === "running");

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">活動ログ</h1>
          <p className="text-sm text-gray-600 mt-1">
            AIチームが「今なにをしているか」「これまで何をしたか」の記録です。新しい順に表示。
          </p>
        </div>

        {/* 今動いているもの */}
        {running.length > 0 && (
          <div className="bg-amber-950/30 border border-amber-800/50 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span className="text-sm font-semibold text-amber-300">いま実施中</span>
            </div>
            {running.map(a => (
              <p key={a.id} className="text-sm text-amber-100/80 pl-4">
                {ROLE_STYLE[a.role]?.emoji} {ROLE_STYLE[a.role]?.label ?? a.role} — {a.step}
              </p>
            ))}
          </div>
        )}

        {/* 指示の出し方の案内 */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-sm text-gray-300 font-medium mb-1">📌 指示の出し方</p>
          <p className="text-xs text-gray-500 leading-relaxed">
            チームへの指示・方針は <code className="text-gray-300 bg-gray-800 px-1 py-0.5 rounded">pdca/feedback.md</code> に書き込むと、
            翌営業日のサイクルから反映されます。チーム同士の評価は
            <a href="/review" className="text-green-400 hover:text-green-300"> チームレビュー</a> ページで確認できます。
          </p>
        </div>

        {items.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
            <span className="text-5xl">📝</span>
            <p className="text-lg font-medium text-gray-500">活動ログはまだありません</p>
            <p className="text-sm">次のPDCAサイクル実行時に記録されます</p>
          </div>
        ) : (
          <div className="space-y-8">
            {Array.from(groups.entries()).map(([day, rows]) => (
              <section key={day}>
                <h2 className="text-sm font-bold text-gray-500 mb-3 font-mono">{day}</h2>
                <div>
                  {rows.map(a => <ActivityRow key={a.id} a={a} />)}
                </div>
              </section>
            ))}
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}

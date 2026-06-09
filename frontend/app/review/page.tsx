import type { Metadata } from "next";
import type { ReactElement } from "react";
import { fetchWeeklyReviews } from "@/lib/data";
import type { WeeklyReview } from "@/lib/types";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "チームレビュー — StockSignal",
  description: "AIチーム（FM・Quant・マーケットコンサル・Engineer）の週次相互評価レポート",
};

// ── 小コンポーネント ──────────────────────────────────────────────────────────

function DeltaBadge({ start, end, unit = "%" }: { start: number | null; end: number | null; unit?: string }) {
  if (start == null || end == null) return <span className="text-gray-600">—</span>;
  const delta = Math.round((end - start) * 100) / 100;
  const up    = delta >= 0;
  return (
    <span className={`font-mono text-sm ${up ? "text-green-400" : "text-red-400"}`}>
      {up ? "▲" : "▼"} {Math.abs(delta)}{unit}
    </span>
  );
}

function MetricRow({ label, start, end }: { label: string; start: number | null; end: number | null }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-600 font-mono">{start ?? "?"}%</span>
        <span className="text-gray-700">→</span>
        <span className="text-sm font-mono text-white">{end ?? "?"}%</span>
        <DeltaBadge start={start} end={end} />
      </div>
    </div>
  );
}

function EvalSection({ title, emoji, content }: { title: string; emoji: string; content: string }) {
  // GOODとBADをパースして色分け
  const lines = content.split("\n");
  const rendered: ReactElement[] = [];
  let key = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) { rendered.push(<div key={key++} className="h-2" />); continue; }

    if (trimmed.startsWith("## ") || trimmed.startsWith("# ")) {
      rendered.push(
        <h3 key={key++} className="text-base font-bold text-white mt-4 mb-2">
          {trimmed.replace(/^#+\s*/, "")}
        </h3>
      );
    } else if (/よかった|GOOD|good/i.test(trimmed) && trimmed.startsWith("###")) {
      rendered.push(
        <div key={key++} className="flex items-center gap-1.5 mt-3 mb-1">
          <span className="text-green-400 text-sm">✓</span>
          <span className="text-green-400 text-sm font-semibold">よかった点</span>
        </div>
      );
    } else if (/問題|BAD|bad|お願い|困/i.test(trimmed) && trimmed.startsWith("###")) {
      rendered.push(
        <div key={key++} className="flex items-center gap-1.5 mt-3 mb-1">
          <span className="text-orange-400 text-sm">!</span>
          <span className="text-orange-400 text-sm font-semibold">改善してほしい点</span>
        </div>
      );
    } else if (/助かった/i.test(trimmed) && trimmed.startsWith("###")) {
      rendered.push(
        <div key={key++} className="flex items-center gap-1.5 mt-3 mb-1">
          <span className="text-green-400 text-sm">✓</span>
          <span className="text-green-400 text-sm font-semibold">助かった点</span>
        </div>
      );
    } else if (/来週|お願い/i.test(trimmed) && trimmed.startsWith("###")) {
      rendered.push(
        <div key={key++} className="flex items-center gap-1.5 mt-3 mb-1">
          <span className="text-blue-400 text-sm">→</span>
          <span className="text-blue-400 text-sm font-semibold">来週やってほしいこと</span>
        </div>
      );
    } else if (trimmed.startsWith("-") || trimmed.match(/^\d+\./)) {
      rendered.push(
        <div key={key++} className="flex gap-2 py-0.5 pl-2">
          <span className="text-gray-600 mt-0.5 shrink-0">•</span>
          <span className="text-sm text-gray-300 leading-relaxed">{trimmed.replace(/^[-\d.]+\s*/, "")}</span>
        </div>
      );
    } else if (trimmed.startsWith("**") && trimmed.endsWith("**")) {
      rendered.push(
        <p key={key++} className="text-sm font-semibold text-gray-200 mt-2">{trimmed.replace(/\*\*/g, "")}</p>
      );
    } else {
      rendered.push(
        <p key={key++} className="text-sm text-gray-400 leading-relaxed">{trimmed}</p>
      );
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xl">{emoji}</span>
        <h2 className="text-base font-bold text-white">{title}</h2>
      </div>
      <div>{rendered}</div>
    </div>
  );
}

function ActionItem({ text }: { text: string }) {
  const trimmed = text.trim().replace(/^[-•*]\s*/, "");
  if (!trimmed) return null;

  // "担当者: 内容" の形式をパース
  const colonIdx = trimmed.indexOf(":");
  if (colonIdx > 0 && colonIdx < 20) {
    const role    = trimmed.slice(0, colonIdx).trim();
    const content = trimmed.slice(colonIdx + 1).trim();
    const colorMap: Record<string, string> = {
      FM:         "bg-blue-900/40 text-blue-300 border-blue-800",
      Quant:      "bg-purple-900/40 text-purple-300 border-purple-800",
      Consultant: "bg-yellow-900/40 text-yellow-300 border-yellow-800",
      Engineer:   "bg-green-900/40 text-green-300 border-green-800",
      Human:      "bg-pink-900/40 text-pink-300 border-pink-800",
    };
    const cls = colorMap[role] ?? "bg-gray-800 text-gray-300 border-gray-700";
    return (
      <div className="flex items-start gap-3 py-3 border-b border-gray-800 last:border-0">
        <span className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded border ${cls}`}>{role}</span>
        <span className="text-sm text-gray-300 leading-relaxed">{content}</span>
      </div>
    );
  }
  return (
    <div className="flex gap-2 py-2 border-b border-gray-800 last:border-0">
      <span className="text-gray-600 shrink-0 mt-0.5">•</span>
      <span className="text-sm text-gray-300 leading-relaxed">{trimmed}</span>
    </div>
  );
}

function ReviewCard({ review }: { review: WeeklyReview }) {
  const actionLines = review.next_actions.split("\n").filter(l => l.trim());

  return (
    <div className="space-y-6">
      {/* 数字サマリー */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-base font-bold text-white mb-4">今週の数字</h2>
        <MetricRow label="平均リターン（21日）"     start={review.avg_start} end={review.avg_end} />
        <MetricRow label="勝率（プラスで終わった率）" start={review.win_start} end={review.win_end} />
        <MetricRow label="大勝率（+8%以上）"        start={review.big_start} end={review.big_end} />
        <div className="flex gap-6 mt-4 pt-3 border-t border-gray-800">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">{review.adopted}</div>
            <div className="text-xs text-gray-600 mt-0.5">採用</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">{review.rejected}</div>
            <div className="text-xs text-gray-600 mt-0.5">却下</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-400">{review.skipped}</div>
            <div className="text-xs text-gray-600 mt-0.5">スキップ</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">{review.signals}</div>
            <div className="text-xs text-gray-600 mt-0.5">買いシグナル</div>
          </div>
        </div>
      </div>

      {/* 各ロールの評価 */}
      <EvalSection title="Engineer からの評価（Quant・FM へ）"    emoji="🔧" content={review.engineer_eval} />
      <EvalSection title="Quant からの評価（Engineer・FM へ）"    emoji="📐" content={review.quant_eval} />
      <EvalSection title="マーケットコンサルからの評価（FM・Quant へ）" emoji="🔍" content={review.securities_eval} />
      <EvalSection title="FM からの評価（Quant・Engineer へ）"     emoji="💼" content={review.fm_eval} />

      {/* Human へのフィードバック */}
      <EvalSection title="AIチームからオーナーへ" emoji="💬" content={review.human_feedback} />

      {/* 来週のアクション */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5">
        <h2 className="text-base font-bold text-white mb-4">来週やること</h2>
        {actionLines.map((line, i) => <ActionItem key={i} text={line} />)}
      </div>
    </div>
  );
}

// ── メインページ ──────────────────────────────────────────────────────────────

export default async function ReviewPage() {
  const reviews = await fetchWeeklyReviews(8);
  const latest  = reviews[0] ?? null;

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">週次チームレビュー</h1>
          <p className="text-sm text-gray-600 mt-1">
            AIチームが互いの仕事ぶりを評価し、来週の改善点をまとめます。毎週月曜更新。
          </p>
        </div>

        {/* 週選択タブ */}
        {reviews.length > 1 && (
          <div className="flex gap-2 flex-wrap">
            {reviews.map((r) => (
              <span
                key={r.week}
                className={`text-xs px-3 py-1.5 rounded-full border font-mono ${
                  r.week === latest?.week
                    ? "bg-green-900/40 border-green-700 text-green-300"
                    : "bg-gray-900 border-gray-700 text-gray-500"
                }`}
              >
                {r.week}
              </span>
            ))}
          </div>
        )}

        {latest ? (
          <>
            <p className="text-xs text-gray-700 font-mono">
              {latest.week} — {new Date(latest.created_at).toLocaleDateString("ja-JP", { month: "long", day: "numeric" })} 生成
            </p>
            <ReviewCard review={latest} />
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
            <span className="text-5xl">📋</span>
            <p className="text-lg font-medium text-gray-500">レビューデータはまだありません</p>
            <p className="text-sm">毎週月曜に自動生成されます</p>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}

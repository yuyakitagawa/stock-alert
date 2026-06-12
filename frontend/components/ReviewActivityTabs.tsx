"use client";
import { useState, type ReactElement } from "react";
import type { WeeklyReview, Activity } from "@/lib/types";

// ─── 活動ログ ────────────────────────────────────────────────────────────────

const ROLE_STYLE: Record<string, { label: string; cls: string; emoji: string }> = {
  FM:         { label: "ファンドマネージャー", cls: "bg-blue-900/40 text-blue-300 border-blue-800",     emoji: "💼" },
  Quant:      { label: "数量アナリスト",       cls: "bg-purple-900/40 text-purple-300 border-purple-800", emoji: "📐" },
  Consultant: { label: "相場リスク管制官",     cls: "bg-yellow-900/40 text-yellow-300 border-yellow-800", emoji: "🚦" },
  Engineer:   { label: "エンジニア",           cls: "bg-green-900/40 text-green-300 border-green-800",   emoji: "🔧" },
  QA:         { label: "品質保証（QA）",       cls: "bg-cyan-900/40 text-cyan-300 border-cyan-800",     emoji: "🛡️" },
  Designer:   { label: "デザイナー",           cls: "bg-rose-900/40 text-rose-300 border-rose-800",     emoji: "🎨" },
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
      <span className={`absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full ${isRunning ? "bg-amber-400 animate-pulse" : "bg-gray-700"}`} />
      <div className="flex items-center gap-2 flex-wrap mb-1.5">
        <span className={`text-xs font-bold px-2 py-0.5 rounded border ${role.cls}`}>{role.emoji} {role.label}</span>
        <span className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${status.cls}`}>{status.label}</span>
        <span className="text-[11px] text-gray-600 font-mono ml-auto">{timeLabel(a.ts)}</span>
      </div>
      <p className="text-sm font-medium text-gray-200">{a.step}</p>
      {a.summary && <p className="text-sm text-gray-400 mt-0.5 leading-relaxed">{a.summary}</p>}
      {a.detail && a.detail.trim() && a.detail !== a.summary && (
        <details className="mt-2 group">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300 select-none">詳細を見る ▾</summary>
          <pre className="mt-2 text-xs text-gray-400 bg-gray-950 border border-gray-800 rounded-lg p-3 whitespace-pre-wrap leading-relaxed font-sans max-h-96 overflow-y-auto">{a.detail}</pre>
        </details>
      )}
    </div>
  );
}

// ─── チームレビュー ──────────────────────────────────────────────────────────

function DeltaBadge({ start, end }: { start: number | null; end: number | null }) {
  if (start == null || end == null) return <span className="text-gray-600">—</span>;
  const delta = Math.round((end - start) * 100) / 100;
  const up = delta >= 0;
  return <span className={`font-mono text-sm ${up ? "text-green-400" : "text-red-400"}`}>{up ? "▲" : "▼"} {Math.abs(delta)}%</span>;
}

function MetricRow({ label, start, end }: { label: string; start: number | null; end: number | null }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-600 font-mono">{start ?? "?"}%</span>
        <span className="text-gray-600">→</span>
        <span className="text-sm font-mono text-white">{end ?? "?"}%</span>
        <DeltaBadge start={start} end={end} />
      </div>
    </div>
  );
}

function EvalSection({ title, emoji, content }: { title: string; emoji: string; content: string }) {
  const lines = content.split("\n");
  const rendered: ReactElement[] = [];
  let key = 0;
  for (const line of lines) {
    const t = line.trim();
    if (!t) { rendered.push(<div key={key++} className="h-2" />); continue; }
    if (t.startsWith("## ") || t.startsWith("# ")) {
      rendered.push(<h3 key={key++} className="text-base font-bold text-white mt-4 mb-2">{t.replace(/^#+\s*/, "")}</h3>);
    } else if (/よかった|GOOD|good/i.test(t) && t.startsWith("###")) {
      rendered.push(<div key={key++} className="flex items-center gap-1.5 mt-3 mb-1"><span className="text-green-400 text-sm">✓</span><span className="text-green-400 text-sm font-semibold">よかった点</span></div>);
    } else if (/問題|BAD|bad|お願い|困/i.test(t) && t.startsWith("###")) {
      rendered.push(<div key={key++} className="flex items-center gap-1.5 mt-3 mb-1"><span className="text-orange-400 text-sm">!</span><span className="text-orange-400 text-sm font-semibold">改善してほしい点</span></div>);
    } else if (/助かった/i.test(t) && t.startsWith("###")) {
      rendered.push(<div key={key++} className="flex items-center gap-1.5 mt-3 mb-1"><span className="text-green-400 text-sm">✓</span><span className="text-green-400 text-sm font-semibold">助かった点</span></div>);
    } else if (/来週|お願い/i.test(t) && t.startsWith("###")) {
      rendered.push(<div key={key++} className="flex items-center gap-1.5 mt-3 mb-1"><span className="text-blue-400 text-sm">→</span><span className="text-blue-400 text-sm font-semibold">来週やってほしいこと</span></div>);
    } else if (t.startsWith("-") || t.match(/^\d+\./)) {
      rendered.push(<div key={key++} className="flex gap-2 py-0.5 pl-2"><span className="text-gray-600 mt-0.5 shrink-0">•</span><span className="text-sm text-gray-300 leading-relaxed">{t.replace(/^[-\d.]+\s*/, "")}</span></div>);
    } else if (t.startsWith("**") && t.endsWith("**")) {
      rendered.push(<p key={key++} className="text-sm font-semibold text-gray-200 mt-2">{t.replace(/\*\*/g, "")}</p>);
    } else {
      rendered.push(<p key={key++} className="text-sm text-gray-400 leading-relaxed">{t}</p>);
    }
  }
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-4"><span className="text-xl">{emoji}</span><h2 className="text-base font-bold text-white">{title}</h2></div>
      <div>{rendered}</div>
    </div>
  );
}

function ActionItem({ text }: { text: string }) {
  const trimmed = text.trim().replace(/^[-•*]\s*/, "");
  if (!trimmed) return null;
  const colonIdx = trimmed.indexOf(":");
  if (colonIdx > 0 && colonIdx < 20) {
    const role = trimmed.slice(0, colonIdx).trim();
    const content = trimmed.slice(colonIdx + 1).trim();
    const colorMap: Record<string, string> = {
      FM: "bg-blue-900/40 text-blue-300 border-blue-800",
      Quant: "bg-purple-900/40 text-purple-300 border-purple-800",
      Consultant: "bg-yellow-900/40 text-yellow-300 border-yellow-800",
      Engineer: "bg-green-900/40 text-green-300 border-green-800",
      Human: "bg-pink-900/40 text-pink-300 border-pink-800",
    };
    return (
      <div className="flex items-start gap-3 py-3 border-b border-gray-800 last:border-0">
        <span className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded border ${colorMap[role] ?? "bg-gray-800 text-gray-300 border-gray-700"}`}>{role}</span>
        <span className="text-sm text-gray-300 leading-relaxed">{content}</span>
      </div>
    );
  }
  return <div className="flex gap-2 py-2 border-b border-gray-800 last:border-0"><span className="text-gray-600 shrink-0 mt-0.5">•</span><span className="text-sm text-gray-300 leading-relaxed">{trimmed}</span></div>;
}

function ReviewCard({ review }: { review: WeeklyReview }) {
  const actionLines = review.next_actions.split("\n").filter(l => l.trim());
  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-base font-bold text-white mb-4">今週の数字</h2>
        <MetricRow label="平均リターン（21日）"      start={review.avg_start} end={review.avg_end} />
        <MetricRow label="勝率（プラスで終わった率）" start={review.win_start} end={review.win_end} />
        <MetricRow label="大勝率（+8%以上）"         start={review.big_start} end={review.big_end} />
        <div className="flex gap-6 mt-4 pt-3 border-t border-gray-800">
          {[
            { v: review.adopted,  label: "採用",        cls: "text-green-400" },
            { v: review.rejected, label: "却下",        cls: "text-red-400" },
            { v: review.skipped,  label: "スキップ",    cls: "text-gray-400" },
            { v: review.signals,  label: "買いシグナル", cls: "text-blue-400" },
          ].map(({ v, label, cls }) => (
            <div key={label} className="text-center">
              <div className={`text-2xl font-bold ${cls}`}>{v}</div>
              <div className="text-xs text-gray-600 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      </div>
      <EvalSection title="Engineer からの評価（Quant・FM へ）"           emoji="🔧" content={review.engineer_eval} />
      <EvalSection title="Quant からの評価（Engineer・FM へ）"           emoji="📐" content={review.quant_eval} />
      <EvalSection title="マーケットコンサルからの評価（FM・Quant へ）" emoji="🔍" content={review.securities_eval} />
      <EvalSection title="FM からの評価（Quant・Engineer へ）"           emoji="💼" content={review.fm_eval} />
      <EvalSection title="AIチームからオーナーへ"                        emoji="💬" content={review.human_feedback} />
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5">
        <h2 className="text-base font-bold text-white mb-4">来週やること</h2>
        {actionLines.map((line, i) => <ActionItem key={i} text={line} />)}
      </div>
    </div>
  );
}

// ─── タブ統合コンポーネント ──────────────────────────────────────────────────

interface Props {
  reviews:    WeeklyReview[];
  activities: Activity[];
}

export default function ReviewActivityTabs({ reviews, activities }: Props) {
  const [tab, setTab] = useState<"review" | "activity">("activity");
  const latest  = reviews[0] ?? null;
  const running = activities.filter(a => a.status === "running");

  const groups = new Map<string, Activity[]>();
  for (const a of activities) {
    const day = a.run_date || a.ts.slice(0, 10);
    if (!groups.has(day)) groups.set(day, []);
    groups.get(day)!.push(a);
  }

  return (
    <div className="space-y-6">
      {/* タブ */}
      <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-lg p-1 w-fit">
        {([
          { key: "activity", label: "📋 活動ログ" },
          { key: "review",   label: "📊 チームレビュー" },
        ] as const).map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              tab === key
                ? "bg-gray-700 text-white"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* 活動ログ */}
      {tab === "activity" && (
        <div className="space-y-6">
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

          {activities.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
              <span className="text-5xl">📝</span>
              <p className="text-lg font-medium text-gray-500">活動ログはまだありません</p>
            </div>
          ) : (
            <div className="space-y-8">
              {Array.from(groups.entries()).map(([day, rows]) => (
                <section key={day}>
                  <h2 className="text-sm font-bold text-gray-500 mb-3 font-mono">{day}</h2>
                  <div>{rows.map(a => <ActivityRow key={a.id} a={a} />)}</div>
                </section>
              ))}
            </div>
          )}
        </div>
      )}

      {/* チームレビュー */}
      {tab === "review" && (
        <div className="space-y-6">
          {reviews.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              {reviews.map(r => (
                <span key={r.week} className={`text-xs px-3 py-1.5 rounded-full border font-mono ${
                  r.week === latest?.week
                    ? "bg-green-900/40 border-green-700 text-green-300"
                    : "bg-gray-900 border-gray-700 text-gray-500"
                }`}>{r.week}</span>
              ))}
            </div>
          )}
          {latest ? (
            <>
              <p className="text-xs text-gray-600 font-mono">
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
        </div>
      )}
    </div>
  );
}

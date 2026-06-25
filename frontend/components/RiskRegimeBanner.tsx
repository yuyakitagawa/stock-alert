import type { RiskRegime } from "@/lib/types";

const STYLE: Record<string, { bg: string; border: string; text: string }> = {
  risk_off: { bg: "bg-red-950/30",    border: "border-red-800",    text: "text-red-300" },
  caution:  { bg: "bg-yellow-950/30", border: "border-yellow-800", text: "text-yellow-300" },
  risk_on:  { bg: "bg-green-950/25",  border: "border-green-900",  text: "text-green-300" },
};

export default function RiskRegimeBanner({ risk }: { risk: RiskRegime | null }) {
  if (!risk) return null;
  const s = STYLE[risk.regime] ?? STYLE.risk_on;

  return (
    <div className={`rounded-xl border ${s.border} ${s.bg} px-4 py-3`}>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-base">🚦</span>
        <span className="text-sm font-bold text-white">相場リスク管制官</span>
        <span className={`text-sm font-bold ${s.text}`}>{risk.label}</span>
        <span className="text-xs text-gray-500">→ {risk.action}</span>
        {risk.suppress_buy && (
          <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-red-900/50 text-red-300 border border-red-700 ml-auto">
            本日のS買いは自動見送り中
          </span>
        )}
      </div>
      {risk.reasons?.length > 0 && (
        <p className="text-xs text-gray-500 mt-1.5 leading-relaxed">
          {risk.reasons.join("・")}
        </p>
      )}
      <p className="text-[11px] text-gray-600 mt-1">
        マクロ（日経20日・VIX・ドル円・S&amp;P500）から自動判定。リスクオフ日は買いシグナルを見送ります。
      </p>
    </div>
  );
}

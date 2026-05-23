"use client";
import RecommendBadge from "./RecommendBadge";
import { useLang } from "@/contexts/LanguageContext";

const SIGNALS = {
  ja: [
    { signal: "S買い",        desc: "全条件クリアの最優先買いシグナル" },
    { signal: "A買い",        desc: "上昇確率が高く買い推奨" },
    { signal: "方向感なし",  desc: "上昇・下落どちらでもない中立状態" },
    { signal: "弱気シグナル", desc: "やや下落傾向、慎重に" },
    { signal: "下降シグナル", desc: "強い下落圧力あり" },
  ],
  en: [
    { signal: "S買い",        desc: "All conditions met — top priority buy" },
    { signal: "A買い",        desc: "High rise probability — buy" },
    { signal: "方向感なし",  desc: "Neutral — no clear direction" },
    { signal: "弱気シグナル", desc: "Slight downward trend — caution" },
    { signal: "下降シグナル", desc: "Strong downtrend" },
  ],
};

// S買い / A買い 条件表（条件変更時はここを更新）
const CONDITIONS = {
  ja: [
    { label: "ネットスコア (上昇確率−下落確率)",  s: "17% ≤ net ≤ 24%", a: "net ≥ 6%" },
    { label: "下落確率",                        s: "< 4%",              a: "—" },
    { label: "年率ボラティリティ",               s: "≤ 25%",            a: "≤ 25%" },
    { label: "連続下落日数",                    s: "≤ 3日",             a: "≤ 3日" },
    { label: "60日ドローダウン",                s: "≥ −15%",           a: "≥ −15%" },
    { label: "流動性 (20日平均売買代金)",        s: "≥ 5,000万円/日",   a: "≥ 5,000万円/日" },
    { label: "次回決算まで",                    s: "22日以上先",        a: "—" },
    { label: "1日の発令件数",                   s: "上位3件まで",       a: "制限なし" },
    { label: "対応ETF前日リターン (XLK/XLF/XLI/XLB/XLV/XLY)", s: "プラスのみ", a: "プラスのみ" },
  ],
  en: [
    { label: "Net Score (Rise prob − Drop prob)",      s: "17% ≤ net ≤ 24%",    a: "net ≥ 6%" },
    { label: "Drop probability",                       s: "< 4%",                a: "—" },
    { label: "Annualized volatility",                  s: "≤ 25%",               a: "≤ 25%" },
    { label: "Consecutive down days",                  s: "≤ 3 days",            a: "≤ 3 days" },
    { label: "60-day drawdown",                        s: "≥ −15%",              a: "≥ −15%" },
    { label: "Liquidity (20-day avg turnover)",        s: "≥ ¥50M/day",          a: "≥ ¥50M/day" },
    { label: "Days until next earnings",               s: "≥ 22 days",           a: "—" },
    { label: "Signals per day",                        s: "Top 3 only",          a: "No limit" },
    { label: "US sector ETF prev-day return (XLK/XLF/XLI/XLB/XLV/XLY)", s: "Positive only", a: "Positive only" },
  ],
};

export default function SignalLegend() {
  const { lang } = useLang();
  const sigs = SIGNALS[lang];
  const conds = CONDITIONS[lang];

  return (
    <div className="border-t border-gray-800/60 pt-8 pb-6 px-4 sm:px-6">
      <div className="max-w-7xl mx-auto space-y-5">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wide">
          {lang === "ja" ? "シグナル凡例" : "Signal Legend"}
        </h3>

        {/* Signal list */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2">
          {sigs.map(({ signal, desc }) => (
            <div key={signal} className="flex items-center gap-3 text-xs">
              <div className="shrink-0">
                <RecommendBadge value={signal} />
              </div>
              <span className="text-gray-500">{desc}</span>
            </div>
          ))}
        </div>

        {/* Buy signal conditions table */}
        <div className="overflow-x-auto rounded-lg border border-gray-800/60">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800/60 bg-gray-900/60">
                <th className="text-left px-3 py-2 text-gray-500 font-medium">
                  {lang === "ja" ? "条件" : "Condition"}
                </th>
                <th className="text-center px-3 py-2 font-medium">
                  <span className="inline-flex items-center gap-1">
                    <RecommendBadge value="S買い" />
                  </span>
                </th>
                <th className="text-center px-3 py-2 font-medium">
                  <span className="inline-flex items-center gap-1">
                    <RecommendBadge value="A買い" />
                  </span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/40">
              {conds.map(({ label, s, a }) => (
                <tr key={label} className="hover:bg-gray-800/20 transition-colors">
                  <td className="px-3 py-2 text-gray-500">{label}</td>
                  <td className="px-3 py-2 text-center text-gray-300 font-mono">{s}</td>
                  <td className="px-3 py-2 text-center text-gray-500 font-mono">{a}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Stop-loss explanation */}
        <div className="bg-red-950/20 border border-red-900/25 rounded-lg px-4 py-3 text-xs text-gray-400 leading-relaxed">
          <span className="text-red-400 font-semibold">
            {lang === "ja" ? "損切りライン" : "Stop-Loss Line"}
          </span>
          {lang === "ja"
            ? " — 購入後に保有し続けるべき下限価格の目安。このラインを割り込んだ場合は損失を確定させることで更なる損失拡大を防ぎます。AIが過去のボラティリティから算出します。"
            : " — The minimum price level you should hold. If the stock falls below this line, it is recommended to sell and cut losses before they deepen. Calculated by AI from historical volatility."}
        </div>

        {/* Prediction logic brief */}
        <p className="text-xs text-gray-600 leading-relaxed">
          {lang === "ja"
            ? "予測ロジック: AIモデル（XGBoost）が34の特徴量から63日後の株価変動確率を予測。ネットスコア（上昇確率－下落確率）でシグナルを判定。S買いはさらに米国セクターETF（XLK/XLF/XLI等）の前日リターンがプラスの時のみ発動（リードラグ効果）。"
            : "Logic: XGBoost AI predicts the probability of 63-day price movement using 34 features. Signals are determined by Net Score (Rise Prob − Drop Prob). S買い additionally requires the prior-day US sector ETF return to be positive (lead-lag effect)."}
        </p>
      </div>
    </div>
  );
}

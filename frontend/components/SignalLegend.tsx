"use client";
import RecommendBadge from "./RecommendBadge";
import { useLang } from "@/contexts/LanguageContext";

const SIGNALS = {
  ja: [
    { signal: "S買い",        desc: "ネットスコア最高 + 下落確率<4% + 米国ETFフィルター通過" },
    { signal: "A買い",        desc: "上昇確率が高く買い推奨" },
    { signal: "方向感なし",  desc: "上昇・下落どちらでもない中立状態" },
    { signal: "弱気シグナル", desc: "やや下落傾向、慎重に" },
    { signal: "下降シグナル", desc: "強い下落圧力あり" },
  ],
  en: [
    { signal: "S買い",        desc: "Top net score + drop prob <4% + US sector ETF filter passed" },
    { signal: "A買い",        desc: "High rise probability — buy" },
    { signal: "方向感なし",  desc: "Neutral — no clear direction" },
    { signal: "弱気シグナル", desc: "Slight downward trend — caution" },
    { signal: "下降シグナル", desc: "Strong downtrend" },
  ],
};

export default function SignalLegend() {
  const { lang } = useLang();
  const sigs = SIGNALS[lang];

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

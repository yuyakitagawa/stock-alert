/**
 * AI運用モデルの詳細ページ。
 *
 * ⚠️ このページの数値・条件はコードのミラー。ロジック/フィルターを変更したら
 *    必ず同じコミットでここも更新すること（CLAUDE.md §7 で厳守ルール化済み）。
 *    真実源（source of truth）:
 *    - 特徴量重要度・AUC・61次元内訳 … feature_importance.json / lib/utils.py(extract_features)
 *    - 品質フィルター            … core/rank_stocks.py passes_buy_filter
 *    - 💎買い条件               … lib/utils.py recommend_from_scores
 *    - 売り判定の段階・推奨ラベル  … lib/utils.py recommend_from_net + rank_stocks.py 判定閾値
 *    - 損切り式・レジーム調整      … core/rank_stocks.py
 */
import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "AI運用モデルの詳細 — StockSignal",
  description:
    "予測モデルの素性と重要度、買いフィルター、売りロジックを公開。XGBoostによる63日先±15%の上昇/下落確率予測の中身を解説。",
};

/* ── 特徴量重要度（feature_importance.json 上位／2026 再学習時点）──────── */
const RISE_FEATURES: { name: string; imp: number; group: string }[] = [
  { name: "USD/JPY 5日変動", imp: 15.3, group: "マクロ" },
  { name: "季節性(sin・月周期)", imp: 12.3, group: "季節性" },
  { name: "配当権利落ち経過日数", imp: 8.2, group: "ファンダ" },
  { name: "S&P500 20日リターン", imp: 7.7, group: "マクロ" },
  { name: "S&P500 5日リターン", imp: 7.1, group: "マクロ" },
  { name: "VIX恐怖指数", imp: 6.4, group: "マクロ" },
  { name: "日経比5日リターン", imp: 6.4, group: "相対強度" },
  { name: "5日リターン", imp: 6.0, group: "テクニカル" },
  { name: "モメンタム加速度", imp: 2.4, group: "テクニカル" },
  { name: "アルファモメンタム", imp: 2.3, group: "相対強度" },
];

const DROP_FEATURES: { name: string; imp: number; group: string }[] = [
  { name: "USD/JPY 5日変動", imp: 10.6, group: "マクロ" },
  { name: "ボラのセクター内相対", imp: 8.4, group: "断面" },
  { name: "季節性(sin・月周期)", imp: 7.0, group: "季節性" },
  { name: "配当権利落ち経過日数", imp: 6.6, group: "ファンダ" },
  { name: "S&P500 20日リターン", imp: 5.5, group: "マクロ" },
  { name: "VIX恐怖指数", imp: 5.0, group: "マクロ" },
  { name: "20日ボラティリティ", imp: 4.2, group: "テクニカル" },
  { name: "S&P500 5日リターン", imp: 3.9, group: "マクロ" },
  { name: "60日ボラティリティ", imp: 3.4, group: "テクニカル" },
  { name: "季節性(cos・月周期)", imp: 3.1, group: "季節性" },
];

const GROUP_COLOR: Record<string, string> = {
  相対強度: "text-sky-400",
  マクロ: "text-amber-400",
  テクニカル: "text-emerald-400",
  ファンダ: "text-violet-400",
  季節性: "text-rose-400",
  断面: "text-teal-400",
};

function FeatureBars({
  title,
  features,
  auc,
  barColor,
}: {
  title: string;
  features: { name: string; imp: number; group: string }[];
  auc: string;
  barColor: string;
}) {
  const max = Math.max(...features.map((f) => f.imp));
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-bold text-white">{title}</h3>
        <span className="text-xs text-gray-500 tabular-nums">AUC {auc}</span>
      </div>
      <ul className="space-y-2">
        {features.map((f) => (
          <li key={f.name} className="text-xs">
            <div className="flex items-baseline justify-between gap-2 mb-0.5">
              <span className="text-gray-300 truncate">
                {f.name}
                <span className={`ml-1.5 ${GROUP_COLOR[f.group] ?? "text-gray-500"}`}>
                  ·{f.group}
                </span>
              </span>
              <span className="text-gray-500 tabular-nums shrink-0">{f.imp.toFixed(1)}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
              <div
                className={`h-full rounded-full ${barColor}`}
                style={{ width: `${(f.imp / max) * 100}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Cond({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-2 text-sm text-gray-300 leading-relaxed">
      <span className="text-green-400 shrink-0 mt-0.5">✓</span>
      <span>{children}</span>
    </li>
  );
}

export default function ModelPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 max-w-3xl mx-auto w-full px-4 sm:px-6 py-8 space-y-10">
        {/* Header */}
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">AI運用モデルの詳細</h1>
          <p className="text-sm text-gray-500 mt-2 leading-relaxed">
            このサイトのシグナルは、<strong className="text-gray-300">XGBoost（勾配ブースティング木）</strong>
            による2つのモデル（上昇／下落）が出力する確率から計算しています。予測対象は
            <strong className="text-gray-300">「63営業日（約3ヶ月）先に株価が±15%動くか」</strong>。
            ネットスコア＝<span className="text-green-400">上昇確率</span> −{" "}
            <span className="text-red-400">下落確率</span> を銘柄選別の中心に据えています。
          </p>
        </div>

        {/* 1. 予測モデルの素性と重要度 */}
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="text-blue-400">1.</span> 予測モデルの素性と重要度
            </h2>
            <p className="text-sm text-gray-500 mt-1.5 leading-relaxed">
              モデルは <strong className="text-gray-300">61次元の特徴量（素性）</strong>を入力します。内訳は
              テクニカル32 ＋ ファンダ11 ＋ マクロ4（VIX・米5日・米20日・USD/JPY）＋ 新規8 ＋ EDINET1 ＋
              断面ランク7。下のバーは、学習済みモデルが実際にどの素性を重視しているか（重要度）を上位10個だけ抜き出したものです。
            </p>
          </div>
          <div className="grid sm:grid-cols-2 gap-3">
            <FeatureBars
              title="上昇モデル（買い方向）"
              features={RISE_FEATURES}
              auc="0.642"
              barColor="bg-emerald-500"
            />
            <FeatureBars
              title="下落モデル（リスク回避）"
              features={DROP_FEATURES}
              auc="0.766"
              barColor="bg-rose-500"
            />
          </div>
          <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 text-sm text-gray-400 leading-relaxed space-y-2">
            <p>
              <strong className="text-gray-200">読み取りポイント：</strong>
              上昇は<span className="text-amber-400">「USD/JPY・VIX・米国株」</span>のマクロ環境と
              <span className="text-rose-400">季節性</span>、<span className="text-violet-400">配当権利落ち</span>が上位を占めます。
              個別銘柄の細かいテクニカルより、<strong className="text-gray-200">地合い</strong>の比重が圧倒的に大きいのが特徴です。
            </p>
            <p>
              下落モデルは<span className="text-amber-400">USD/JPY</span>と<span className="text-teal-400">セクター内ボラ</span>に強く反応し、
              AUC 0.766 と上昇（0.642）より精度が高い。設計思想として
              <strong className="text-gray-200">「上昇を当てる」より「下落を避ける」を優先</strong>しています。
            </p>
          </div>
        </section>

        {/* 2. 買いフィルター */}
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="text-blue-400">2.</span> 買いフィルター
            </h2>
            <p className="text-sm text-gray-500 mt-1.5 leading-relaxed">
              買いは2段構え。まず<strong className="text-gray-300">品質フィルター</strong>で論外な銘柄を落とし、
              通過したものだけが <span className="text-cyan-300 font-semibold">💎 買い</span> 判定の条件審査に進みます。
            </p>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-bold text-gray-200 mb-3">① 品質フィルター（最低ライン）</h3>
            <ul className="space-y-2">
              <Cond>株価 ≥ 300円（低位株を除外）</Cond>
              <Cond>60日ドローダウン ≥ −20%（急落中は除外）</Cond>
              <Cond>連続下落 ≤ 4日（下落トレンド中は除外）</Cond>
              <Cond>RSI &lt; 80（過熱・高値づかみを回避）</Cond>
              <Cond>20日平均売買代金 ≥ 50百万円（流動性確保）</Cond>
            </ul>
          </div>

          <div className="bg-gray-900 border border-cyan-900/50 rounded-xl p-4">
            <h3 className="text-sm font-bold text-cyan-300 mb-1">
              ② 💎 買い 判定（下記を<span className="underline">すべて</span>満たす）
            </h3>
            <p className="text-xs text-gray-500 mb-3">
              極めて厳しく、地合いと業績の両輪が揃った銘柄だけに付く最上位シグナル。
            </p>
            <ul className="space-y-2">
              <Cond>下落確率 &lt; 2%（モデルが下落をほぼ否定）</Cond>
              <Cond>ネットスコア ≥ +16（上昇 − 下落）</Cond>
              <Cond>Piotroskiスコア ≥ 6/9（財務健全性）</Cond>
              <Cond>52週レンジ内位置 &lt; 0.45（高値圏ではない＝伸びしろ）</Cond>
              <Cond>20日ボラティリティ ≤ 20%（値動きが穏やか）</Cond>
              <Cond>90日リターン &gt; −25%（直近で崩れていない）</Cond>
              <Cond>20日平均売買代金 ≥ 100百万円（十分な流動性）</Cond>
              <Cond>業績改善：EPSサプライズ &gt; +2% または BPS成長 &gt; 0</Cond>
            </ul>
          </div>

          <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 text-sm text-gray-400 leading-relaxed">
            <strong className="text-gray-200">さらに後段の降格チェック：</strong>
            💎買いでも、株主優待権利落ち21日前・
            決算テキストの強い悲観（感情スコア ≤ −0.5）・
            強相関の米国セクターETFが前日マイナス・リスクオフ地合い、のいずれかに該当すると
            <span className="text-gray-300">「方向感なし」へ自動降格</span>します。
          </div>
        </section>

        {/* 3. 売りロジック */}
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="text-blue-400">3.</span> 売りロジック
            </h2>
            <p className="text-sm text-gray-500 mt-1.5 leading-relaxed">
              売り側は<strong className="text-gray-300">下落確率の高さ＝ネットスコアのマイナス幅</strong>で段階表示します。
              判定ラベルと推奨は連動します。
            </p>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-800/60 text-gray-400 text-xs">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">ネットスコア</th>
                  <th className="text-left px-4 py-2 font-medium">判定</th>
                  <th className="text-left px-4 py-2 font-medium">推奨</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                <tr>
                  <td className="px-4 py-2 tabular-nums text-gray-300">≥ +15</td>
                  <td className="px-4 py-2">🟢 強気買い</td>
                  <td className="px-4 py-2 text-gray-500">条件次第で 💎買い</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 tabular-nums text-gray-300">+5 〜 +15</td>
                  <td className="px-4 py-2">🔵 やや強気</td>
                  <td className="px-4 py-2 text-gray-500">—</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 tabular-nums text-gray-300">−5 〜 +5</td>
                  <td className="px-4 py-2">🟡 中立</td>
                  <td className="px-4 py-2 text-gray-500">⏳ 方向感なし</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 tabular-nums text-gray-300">−10 〜 −5</td>
                  <td className="px-4 py-2">🟠 やや弱気</td>
                  <td className="px-4 py-2 text-amber-400">⚠️ 弱気シグナル</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 tabular-nums text-gray-300">&lt; −10</td>
                  <td className="px-4 py-2">🔴 売り検討</td>
                  <td className="px-4 py-2 text-red-400">🔴 下降シグナル</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-bold text-gray-200 mb-3">損切りライン（保有後の出口）</h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              各銘柄に <strong className="text-gray-300">1.5 ATR 相当の損切り価格</strong>を自動算出して表示します。
              計算式は <code className="text-xs bg-gray-800 px-1.5 py-0.5 rounded text-gray-300">
              損切り = 株価 × (1 − 1.5 × 20日ボラ × √(20/252))</code>。
              ボラが大きい銘柄ほど損切り幅は広く取り、ダマシでの早すぎる撤退を防ぎます。
            </p>
          </div>

          <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 text-sm text-gray-400 leading-relaxed">
            <strong className="text-gray-200">地合いによる全体防御：</strong>
            日経20日リターンが閾値を下回る下落相場では、推奨銘柄数を 強気10→中立5→弱気3 へ動的に縮小。
            VIX &gt; 30 の恐怖相場ではさらに −1 し、リスクオフ判定時は 💎買いを全件見送ります。
            <span className="text-gray-300">「買わない」も立派な売りシグナル</span>という設計です。
          </div>
        </section>

        {/* Disclaimer */}
        <p className="text-xs text-gray-600 leading-relaxed border-t border-gray-800 pt-4">
          ※ 本ページの数値（特徴量重要度・AUC等）は直近の再学習時点のものです。モデルは毎週金曜に再学習され、重要度は変動します。
          掲載内容は投資判断の参考情報であり、特定銘柄の売買を推奨・保証するものではありません。最終的な投資判断はご自身の責任で行ってください。
        </p>
      </main>
      <Footer />
    </div>
  );
}

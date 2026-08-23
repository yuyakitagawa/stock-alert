// 銘柄カード用の業種アイコン。会社ロゴは取得元・商標の問題で使えないため、
// JPX33業種（jpx_stock_list.sector）を絵文字アイコンで代替する。
// サーバー・クライアントどちらのコンポーネントからも使えるよう素のspanのみで描く。
const SECTOR_EMOJI: Record<string, string> = {
  "水産・農林業": "🐟",
  鉱業: "⛏️",
  建設業: "🏗️",
  食料品: "🍙",
  繊維製品: "🧵",
  "パルプ・紙": "📄",
  化学: "🧪",
  医薬品: "💊",
  "石油・石炭製品": "🛢️",
  ゴム製品: "🛞",
  "ガラス・土石製品": "🧱",
  鉄鋼: "⚒️",
  非鉄金属: "🪙",
  金属製品: "🔩",
  機械: "⚙️",
  電気機器: "🔌",
  輸送用機器: "🚗",
  精密機器: "🔬",
  その他製品: "🧩",
  "電気・ガス業": "⚡",
  陸運業: "🚚",
  海運業: "🚢",
  空運業: "✈️",
  "倉庫・運輸関連業": "📦",
  "情報・通信業": "💻",
  卸売業: "🌐",
  小売業: "🛍️",
  銀行業: "🏦",
  "証券、商品先物取引業": "📈",
  保険業: "🛡️",
  その他金融業: "💰",
  不動産業: "🏢",
  サービス業: "🤝",
};

// 業種が未登録・"-"（ETF等）の銘柄はビル既定アイコンではなく汎用の会社アイコンにする。
const FALLBACK_EMOJI = "💼";

const SIZE_CLASSES = {
  sm: "h-5 w-5 text-[11px]",
  md: "h-7 w-7 text-[15px]",
} as const;

export default function SectorIcon({
  sector,
  size = "md",
}: {
  sector?: string | null;
  size?: keyof typeof SIZE_CLASSES;
}) {
  const emoji = (sector && SECTOR_EMOJI[sector]) || FALLBACK_EMOJI;
  return (
    <span
      aria-hidden
      title={sector && SECTOR_EMOJI[sector] ? sector : undefined}
      className={`flex shrink-0 select-none items-center justify-center rounded-full bg-brand-navy/10 leading-none ${SIZE_CLASSES[size]}`}
    >
      {emoji}
    </span>
  );
}

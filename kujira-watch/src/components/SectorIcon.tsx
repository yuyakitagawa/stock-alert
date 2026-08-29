import type { ReactNode } from "react";

// 銘柄カード用の業種アイコン。会社ロゴは取得元・商標の問題で使えないため、
// JPX33業種（jpx_stock_list.sector）を自作のSVGラインアイコンで代替する。
// 絵文字はOS・ブラウザごとに見た目がバラつくため、ブランド色（紺）の線画で統一する。
// サーバー・クライアントどちらのコンポーネントからも使えるよう素のspan+svgのみで描く。
// 各グリフは24x24グリッドのストローク描画（fillは中心点などの一部のみ）。
const SECTOR_GLYPHS: Record<string, ReactNode> = {
  // 魚
  "水産・農林業": (
    <>
      <path d="M3.5 12c2.4-3.2 5.4-4.8 8.5-4.8s5.8 1.8 8 4.8c-2.2 3-4.9 4.8-8 4.8S5.9 15.2 3.5 12z" />
      <path d="M3.5 12L1.8 9.8M3.5 12l-1.7 2.2" />
      <circle cx="16.6" cy="10.8" r="0.9" fill="currentColor" stroke="none" />
    </>
  ),
  // つるはし
  鉱業: (
    <>
      <path d="M4.5 19.5L15 9" />
      <path d="M9.5 3.5C13 6 18 11 20.5 14.5" />
      <path d="M9.5 3.5C16 3 21 8 20.5 14.5" />
    </>
  ),
  // クレーン
  建設業: (
    <>
      <path d="M7 20.5V5h13" />
      <path d="M4 20.5h6" />
      <path d="M7 5L3.5 8.5" />
      <path d="M16.5 5v4.5" />
      <circle cx="16.5" cy="11" r="1.4" />
    </>
  ),
  // 茶碗と湯気
  食料品: (
    <>
      <path d="M4 12.5a8 8 0 0016 0z" />
      <path d="M8.5 20.5h7" />
      <path d="M9.5 8.5v-2M12 9V6M14.5 8.5v-2" />
    </>
  ),
  // Tシャツ
  繊維製品: (
    <path d="M7.8 4.5L3.5 7l2.2 3.6 2.1-1.2v10.1h8.4V9.4l2.1 1.2L20.5 7l-4.3-2.5a4.2 4.2 0 01-8.4 0z" />
  ),
  // 書類
  "パルプ・紙": (
    <>
      <path d="M6.5 3.5h7.5l4 4v13H6.5z" />
      <path d="M14 3.5V8h4" />
    </>
  ),
  // フラスコ
  化学: (
    <>
      <path d="M10 3.5v5.5L4.8 18a1.6 1.6 0 001.4 2.5h11.6a1.6 1.6 0 001.4-2.5L14 9V3.5" />
      <path d="M8.5 3.5h7" />
      <path d="M7.2 15h9.6" />
    </>
  ),
  // カプセル
  医薬品: (
    <g transform="rotate(-30 12 12)">
      <rect x="4.5" y="8.9" width="15" height="6.2" rx="3.1" />
      <path d="M12 8.9v6.2" />
    </g>
  ),
  // しずく（油滴）
  "石油・石炭製品": (
    <path d="M12 3.5s5.8 6.8 5.8 10.4a5.8 5.8 0 11-11.6 0C6.2 10.3 12 3.5 12 3.5z" />
  ),
  // タイヤ
  ゴム製品: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3.2" />
      <path d="M12 4v2.6M12 17.4V20M4 12h2.6M17.4 12H20" />
    </>
  ),
  // レンガ壁
  "ガラス・土石製品": (
    <>
      <rect x="3" y="6.5" width="18" height="11" />
      <path d="M3 12h18" />
      <path d="M12 6.5V12M8 12v5.5M16 12v5.5" />
    </>
  ),
  // H形鋼
  鉄鋼: (
    <>
      <path d="M6.5 5h11M6.5 19h11" />
      <path d="M12 5v14" />
      <path d="M9 5v2.5M15 5v2.5M9 16.5V19M15 16.5V19" />
    </>
  ),
  // インゴット（金属塊）
  非鉄金属: (
    <>
      <path d="M4 18.5l2-4.5h12l2 4.5z" />
      <path d="M8 14l1.7-4h4.6L16 14" />
    </>
  ),
  // ナット
  金属製品: (
    <>
      <path d="M12 3.5l7.4 4.25v8.5L12 20.5l-7.4-4.25v-8.5z" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  // 歯車
  機械: (
    <>
      <circle cx="12" cy="12" r="5" />
      <circle cx="12" cy="12" r="1.7" />
      <path d="M12 3.5V7M12 17v3.5M3.5 12H7M17 12h3.5M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18" />
    </>
  ),
  // 電源プラグ
  電気機器: (
    <>
      <path d="M9.2 3v4.5M14.8 3v4.5" />
      <path d="M6.5 7.5h11v3.2a5.5 5.5 0 01-11 0z" />
      <path d="M12 16.2V21" />
    </>
  ),
  // 自動車
  輸送用機器: (
    <>
      <path d="M3 16.5v-3L5.8 7h12.4L21 13.5v3" />
      <path d="M3 13.5h18" />
      <circle cx="7.3" cy="17.3" r="1.8" />
      <circle cx="16.7" cy="17.3" r="1.8" />
    </>
  ),
  // 照準（計測）
  精密機器: (
    <>
      <circle cx="12" cy="12" r="7" />
      <path d="M12 3.5V6.7M12 17.3v3.2M3.5 12h3.2M17.3 12h3.2" />
      <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" />
    </>
  ),
  // 立方体（プロダクト）
  その他製品: (
    <>
      <path d="M12 3l8 4.6v8.8L12 21l-8-4.6V7.6z" />
      <path d="M12 12.2l8-4.6M12 12.2L4 7.6M12 12.2V21" />
    </>
  ),
  // 稲妻
  "電気・ガス業": <path d="M13.2 2.5L5 13.5h5.6L9.4 21.5l8.3-11h-5.7z" />,
  // トラック
  陸運業: (
    <>
      <rect x="2.5" y="7" width="12" height="8.5" />
      <path d="M14.5 10h3.6l3.4 3.5v2h-2.7" />
      <circle cx="7" cy="17.6" r="1.8" />
      <circle cx="16.8" cy="17.6" r="1.8" />
    </>
  ),
  // コンテナ船
  海運業: (
    <>
      <path d="M3 13.5l2.3 5h13.4l2.3-5z" />
      <path d="M7 13.5V8.5h4.5v5" />
      <path d="M11.5 13.5v-3h5v3" />
    </>
  ),
  // 紙飛行機
  空運業: (
    <>
      <path d="M21 3.5L3.2 10.8l6.9 2.6 2.7 6.8z" />
      <path d="M21 3.5L10.1 13.4" />
    </>
  ),
  // 倉庫
  "倉庫・運輸関連業": (
    <>
      <path d="M3.5 20v-9.6L12 5l8.5 5.4V20" />
      <path d="M7.5 20v-5.5h9V20" />
      <path d="M7.5 17.3h9" />
    </>
  ),
  // モニター
  "情報・通信業": (
    <>
      <rect x="3" y="4.5" width="18" height="12" rx="1.5" />
      <path d="M12 16.5v3.5M8.5 20h7" />
    </>
  ),
  // 地球（経線）
  卸売業: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M4.3 12h15.4" />
      <path d="M12 4c3 2.4 3 13.6 0 16-3-2.4-3-13.6 0-16z" />
    </>
  ),
  // ショッピングバッグ
  小売業: (
    <>
      <path d="M5.8 8.2h12.4l-1.1 12.3H6.9z" />
      <path d="M9 11V6.8a3 3 0 016 0V11" />
    </>
  ),
  // 銀行（柱）
  銀行業: (
    <>
      <path d="M3.5 9L12 4l8.5 5z" />
      <path d="M5.5 9v7.5M9.8 9v7.5M14.2 9v7.5M18.5 9v7.5" />
      <path d="M4 16.5h16M3 20h18" />
    </>
  ),
  // 上昇チャート
  "証券、商品先物取引業": (
    <>
      <path d="M3.5 20h17" />
      <path d="M4.5 15.5l4.6-4.6 3.4 3.4 6.5-6.5" />
      <path d="M14.5 7.8H19v4.5" />
    </>
  ),
  // 盾＋チェック
  保険業: (
    <>
      <path d="M12 3l7.5 2.8v5.4c0 4.6-3 8.2-7.5 9.8-4.5-1.6-7.5-5.2-7.5-9.8V5.8z" />
      <path d="M8.8 12l2.3 2.3 4.1-4.4" />
    </>
  ),
  // 硬貨（¥）
  その他金融業: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M9 8l3 3.8L15 8M12 11.8V16.5M9.7 13.2h4.6M9.7 15.2h4.6" />
    </>
  ),
  // ビル
  不動産業: (
    <>
      <rect x="6.5" y="3.5" width="11" height="17" />
      <path d="M9.5 7h1.5M13 7h1.5M9.5 10.5h1.5M13 10.5h1.5M9.5 14h1.5M13 14h1.5" />
      <path d="M10.7 20.5v-3.2h2.6v3.2" />
      <path d="M4.5 20.5h15" />
    </>
  ),
  // 呼び鈴（サービスベル）
  サービス業: (
    <>
      <path d="M4.8 16.5a7.2 7.2 0 0114.4 0z" />
      <path d="M12 9.3V7.8" />
      <circle cx="12" cy="6.6" r="1.1" />
      <path d="M3.5 19.5h17" />
    </>
  ),
};

// 業種が未登録・"-"（ETF等）の銘柄はブリーフケース（汎用の会社アイコン）にする。
const FALLBACK_GLYPH: ReactNode = (
  <>
    <rect x="3.5" y="8" width="17" height="11.5" rx="1.5" />
    <path d="M9.2 8V6.2A2.2 2.2 0 0111.4 4h1.2a2.2 2.2 0 012.2 2.2V8" />
    <path d="M3.5 13.5h17" />
  </>
);

// 業種グループごとの色。33業種すべてに固有色を割り当てても見分けがつかないため、
// 4グループにまとめてブランドカラー（紺・青・金・濃紺）を割り当てる。カードを縦に
// 並べたとき左端に色の列ができるので、スクロールしただけで業種の偏りが読み取れる
// （2026-08-29追加。新しい色は足さず、既存のブランドトークンだけで構成している）。
const SECTOR_GROUP: Record<string, "material" | "maker" | "consumer" | "finance"> = {
  // 資源・素材
  "水産・農林業": "material",
  鉱業: "material",
  "石油・石炭製品": "material",
  化学: "material",
  鉄鋼: "material",
  非鉄金属: "material",
  "ガラス・土石製品": "material",
  "パルプ・紙": "material",
  繊維製品: "material",
  ゴム製品: "material",
  金属製品: "material",
  // ものづくり
  建設業: "maker",
  機械: "maker",
  電気機器: "maker",
  輸送用機器: "maker",
  精密機器: "maker",
  その他製品: "maker",
  // 生活・商業・サービス
  食料品: "consumer",
  医薬品: "consumer",
  小売業: "consumer",
  卸売業: "consumer",
  サービス業: "consumer",
  "情報・通信業": "consumer",
  // 金融・インフラ・運輸
  銀行業: "finance",
  "証券、商品先物取引業": "finance",
  保険業: "finance",
  その他金融業: "finance",
  不動産業: "finance",
  "電気・ガス業": "finance",
  陸運業: "finance",
  海運業: "finance",
  空運業: "finance",
  "倉庫・運輸関連業": "finance",
};

// lg（チップ）用の塗り。地色に業種グループの色を敷き、グリフは紙色で抜く。
const GROUP_CHIP: Record<string, string> = {
  material: "bg-brand-navy text-paper",
  maker: "bg-brand-blue text-paper",
  consumer: "bg-brand-gold text-paper",
  finance: "bg-brand-blue-dark text-paper",
};
// 業種が取れない銘柄（上場銘柄マスターに無い等）は色を持たせない。
const FALLBACK_CHIP = "bg-brand-navy/10 text-brand-navy/70";

const SIZES = {
  sm: { circle: "h-5 w-5 rounded-full", svg: 12, chip: false },
  md: { circle: "h-7 w-7 rounded-full", svg: 16, chip: false },
  // 一覧カード用の大型チップ。16pxの線画はカードの中でほぼ見えておらず、33業種ぶん
  // 自作したSVGが効いていなかったため、40px・角丸・業種グループ色で出す。
  lg: { circle: "h-10 w-10 rounded-xl", svg: 21, chip: true },
} as const;

export default function SectorIcon({
  sector,
  size = "md",
}: {
  sector?: string | null;
  size?: keyof typeof SIZES;
}) {
  const known = Boolean(sector && SECTOR_GLYPHS[sector]);
  const glyph = (sector && SECTOR_GLYPHS[sector]) || FALLBACK_GLYPH;
  const { circle, svg, chip } = SIZES[size];
  const paint = chip
    ? (sector && GROUP_CHIP[SECTOR_GROUP[sector]]) || FALLBACK_CHIP
    : "bg-brand-navy/10 text-brand-navy/80";
  return (
    <span
      aria-hidden
      title={known ? (sector as string) : undefined}
      className={`flex shrink-0 select-none items-center justify-center ${paint} ${circle}`}
    >
      <svg
        viewBox="0 0 24 24"
        width={svg}
        height={svg}
        fill="none"
        stroke="currentColor"
        strokeWidth={chip ? 1.8 : 2}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {glyph}
      </svg>
    </span>
  );
}

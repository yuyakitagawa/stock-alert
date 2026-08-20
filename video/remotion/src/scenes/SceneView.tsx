import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {
  PLATE_BG,
  TEXT_SHADOW,
  accentFor,
  brand,
  contentWidth,
  fontFamily,
  safeArea,
} from '../theme';
import type {Scene, ShortProps} from '../types';

/** ループ再生で頭と繋げるため、ctaの末尾この長さだけ冒頭と同じ金額組版に戻す。 */
const LOOP_TAIL_FRAMES = 24;

/**
 * 1シーンの見た目。kind ごとに中央のビジュアルを切り替え、下段に字幕を大きく出す。
 *
 * 設計方針（2026-08-19のインフルエンサー100人レビューを反映）:
 * - 無音で見る視聴者が多数派なので、字幕は「26字の要約1本」を72pxで出す。
 *   ナレーション全文の同時表示は、実機で読めないうえ音声と競合するため廃止した。
 * - 文字の可読性は影ではなく不透明の下地（PLATE_BG）で担保する。
 * - 表示はすべて safeArea 内。左右は対称で、画面中心と要素の中心が一致する。
 * - カット感を出すため、シーン頭は spring で「飛び込ませる」。フェードは使わない。
 * - シーン内でも2.6秒ごとに微小な拍（マイクロビート）を打ち、完全な静止画を作らない。
 */
export const SceneView: React.FC<{
  scene: Scene;
  props: ShortProps;
  sceneIndex: number;
  durationInFrames: number;
}> = ({scene, props, durationInFrames}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 16, stiffness: 180}});
  const accent = accentFor(props.direction);

  // 2.6秒周期・0.27秒だけ1.00→1.03→1.00。静止フレームを無くすための微小な拍。
  const beat = Math.max(0, 1 - Math.abs((frame % 78) - 4) / 4);

  // ctaの末尾はループの継ぎ目。冒頭と同じ絵に戻すため字幕とヘッダを引く。
  const inLoopTail =
    scene.kind === 'cta' && frame >= durationInFrames - LOOP_TAIL_FRAMES;

  return (
    <AbsoluteFill
      style={{
        fontFamily,
        paddingTop: safeArea.top,
        paddingBottom: safeArea.bottom,
        paddingLeft: safeArea.left,
        paddingRight: safeArea.right,
        justifyContent: 'space-between',
        alignItems: 'stretch',
      }}
    >
      {/* 上段: 銘柄と出典の常時表示。どのシーンから見始めても文脈と一次情報がわかる。
          ループ末尾は冒頭と同じ絵にするため hook として描く */}
      <Header props={props} kind={inLoopTail ? 'hook' : scene.kind} />

      {/* 中段: kindごとのビジュアル */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          transform: `translateY(${interpolate(enter, [0, 1], [56, 0])}px) scale(${
            1 + beat * 0.03
          })`,
          opacity: interpolate(enter, [0, 0.35], [0, 1], {extrapolateRight: 'clamp'}),
        }}
      >
        <Visual
          scene={scene}
          props={props}
          accent={accent}
          durationInFrames={durationInFrames}
        />
      </div>

      {/* 下段: 字幕（無音視聴者のための主役）と、全編出しっぱなしの免責 */}
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14}}>
        <Caption
          text={inLoopTail ? props.scenes[0]?.caption ?? '' : scene.caption}
          accent={accent}
        />
        <Disclaimer />
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ */

/**
 * 銘柄行と出典行。出典（EDINETの書類名と提出日）を常時出すことで、
 * 一次情報に基づく解説であることが途中から見た視聴者にも伝わる。
 */
const Header: React.FC<{props: ShortProps; kind: Scene['kind']}> = ({props, kind}) => {
  // hook と cta は中央の金額が主役なので、銘柄行を出さず出典行だけ薄く残す。
  const showTicker = kind !== 'hook' && kind !== 'cta';
  return (
    <div style={{display: 'flex', flexDirection: 'column', gap: 10}}>
      {showTicker ? (
        <div style={{display: 'flex', alignItems: 'center', gap: 16, height: 60}}>
          <span
            style={{fontSize: 36, fontWeight: 900, color: '#ffffff', textShadow: TEXT_SHADOW}}
          >
            {props.stockName}
          </span>
          <span style={{fontSize: 28, fontWeight: 700, color: brand.cream, opacity: 0.6}}>
            {props.stockCode}
          </span>
          <span
            style={{
              marginLeft: 'auto',
              fontSize: 28,
              fontWeight: 900,
              color: brand.navyDeep,
              background: props.direction === 'buy' ? brand.buy : brand.sell,
              borderRadius: 999,
              padding: '8px 22px',
            }}
          >
            {props.direction === 'buy' ? '買い' : '売り'}
          </span>
        </div>
      ) : (
        <div style={{height: 60}} />
      )}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 22,
          fontWeight: 700,
          color: brand.cream,
          opacity: showTicker ? 0.62 : 0.4,
          textShadow: TEXT_SHADOW,
        }}
      >
        <span>EDINET 大量保有報告書 {props.discDate} 提出</span>
        <span style={{color: brand.goldBright}}>kujira-watch.com</span>
      </div>
    </div>
  );
};

const Caption: React.FC<{text: string; accent: string}> = ({text, accent}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  // 字幕は本文より一拍遅れて入る（視線が中央→下段の順で動くように）
  const enter = spring({frame: frame - 4, fps, config: {damping: 15, stiffness: 200}});
  if (!text) {
    return null;
  }
  return (
    <div
      style={{
        alignSelf: 'center',
        transform: `scale(${interpolate(enter, [0, 1], [0.92, 1])})`,
        opacity: interpolate(enter, [0, 0.4], [0, 1], {extrapolateRight: 'clamp'}),
        background: 'rgba(6,10,20,0.88)',
        border: `3px solid ${accent}`,
        borderRadius: 20,
        padding: '22px 34px',
        maxWidth: contentWidth,
      }}
    >
      <div
        style={{
          fontSize: 68,
          fontWeight: 900,
          lineHeight: 1.25,
          color: '#ffffff',
          textAlign: 'center',
          textShadow: TEXT_SHADOW,
        }}
      >
        {text}
      </div>
    </div>
  );
};

/** 金融解説として全編出しておく免責。CTAだけに出しても大半の視聴者には届かない。 */
const Disclaimer: React.FC = () => (
  <div
    style={{
      fontSize: 24,
      fontWeight: 600,
      color: brand.cream,
      opacity: 0.55,
      textShadow: TEXT_SHADOW,
      textAlign: 'center',
    }}
  >
    ※公開情報の要約です。投資勧誘・投資助言ではありません
  </div>
);

/* ------------------------------------------------------------------ */

const Visual: React.FC<{
  scene: Scene;
  props: ShortProps;
  accent: string;
  durationInFrames: number;
}> = ({scene, props, accent, durationInFrames}) => {
  switch (scene.kind) {
    case 'hook':
      return <HookVisual props={props} accent={accent} />;
    case 'company':
      return <CompanyVisual props={props} />;
    case 'deal':
      return <DealVisual props={props} accent={accent} />;
    case 'filer':
      return <FilerVisual props={props} accent={accent} />;
    case 'change':
      return <ChangeVisual props={props} accent={accent} />;
    case 'chart':
      return <ChartVisual scene={scene} accent={accent} />;
    case 'cta':
      return <CtaVisual props={props} accent={accent} durationInFrames={durationInFrames} />;
    default:
      return null;
  }
};

/**
 * 金額の一撃。冒頭とループ末尾で1pxもずれないよう、両方からこの1つを呼ぶ。
 * 数字は白＋濃い下地で、accent色は単位だけに使う（背景の明部に金色が沈む事故の対策）。
 */
const AmountSlam: React.FC<{amountOku: number; accent: string; scale: number}> = ({
  amountOku,
  accent,
  scale,
}) => (
  <div
    style={{
      background: PLATE_BG,
      borderRadius: 28,
      padding: '10px 40px',
      transform: `scale(${scale})`,
    }}
  >
    <div
      style={{
        fontSize: 210,
        fontWeight: 900,
        lineHeight: 1.05,
        color: '#ffffff',
        // 「億」と「円」の間で折り返さないこと（760pxの枠に収まらず2行になる事故の対策）
        whiteSpace: 'nowrap',
        textShadow: '0 10px 60px rgba(0,0,0,0.6)',
      }}
    >
      {formatAmount(amountOku)}
      <span style={{fontSize: 100, color: accent}}>億円</span>
    </div>
  </div>
);

/**
 * 冒頭。0.35秒で金額を叩き込んだあと、社名→動詞→提出者→保有比率の順に
 * 約2.7秒かけて4回情報を足す。静止したまま12秒待たせる作りをやめた（2026-08-19）。
 */
const HookVisual: React.FC<{
  props: ShortProps;
  accent: string;
  /** ループ末尾から呼ぶとき用。演出を飛ばして完成形だけを描く */
  revealed?: boolean;
}> = ({props, accent, revealed = false}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const slam = revealed ? 1 : spring({frame, fps, config: {damping: 12, stiffness: 320}});
  const verb = props.direction === 'buy' ? '買い集めた' : '売り払った';
  const pill = revealed ? 1 : spring({frame: frame - 40, fps, config: {damping: 14, stiffness: 220}});
  const at = (a: number, b: number) =>
    revealed
      ? 1
      : interpolate(frame, [a, b], [0, 1], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        });

  return (
    <div style={{textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6}}>
      <div
        style={{
          fontSize: 60,
          fontWeight: 900,
          color: '#ffffff',
          textShadow: TEXT_SHADOW,
          opacity: at(6, 12),
        }}
      >
        {props.stockName}を
      </div>
      <AmountSlam
        amountOku={props.dealAmountOku}
        accent={accent}
        scale={interpolate(slam, [0, 1], [1.35, 1])}
      />
      <div
        style={{
          fontSize: 72,
          fontWeight: 900,
          color: '#ffffff',
          textShadow: TEXT_SHADOW,
          opacity: at(20, 26),
        }}
      >
        {verb}
      </div>
      {/* 3拍目: 誰が買ったのか（一番引きの強い情報を下段captionから中央へ引き上げる） */}
      <div
        style={{
          marginTop: 14,
          fontSize: 44,
          fontWeight: 900,
          color: brand.navyDeep,
          background: accent,
          borderRadius: 999,
          padding: '12px 34px',
          transform: `translateY(${interpolate(pill, [0, 1], [48, 0])}px)`,
          opacity: interpolate(pill, [0, 0.4], [0, 1], {extrapolateRight: 'clamp'}),
        }}
      >
        {props.dealTypeLabel}
      </div>
      {/* 4拍目: 規模の裏付け */}
      <div
        style={{
          marginTop: 10,
          fontSize: 52,
          fontWeight: 900,
          color: '#ffffff',
          textShadow: TEXT_SHADOW,
          opacity: at(70, 80),
        }}
      >
        保有比率 {props.holdingRatio.toFixed(2)}%
      </div>
    </div>
  );
};

const CompanyVisual: React.FC<{props: ShortProps}> = ({props}) => (
  <Card>
    <Label text="どんな会社？" color={brand.goldBright} />
    <div
      style={{
        background: PLATE_BG,
        borderRadius: 24,
        padding: '26px 34px',
      }}
    >
      <div style={{fontSize: 84, fontWeight: 900, color: '#ffffff', lineHeight: 1.2}}>
        {props.stockName}
      </div>
      <div style={{fontSize: 40, color: brand.cream, opacity: 0.7, marginTop: 8}}>
        証券コード {props.stockCode}
      </div>
    </div>
  </Card>
);

const DealVisual: React.FC<{props: ShortProps; accent: string}> = ({props, accent}) => {
  const frame = useCurrentFrame();
  const countProgress = interpolate(frame, [4, 34], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const shown = props.dealAmountOku * (1 - Math.pow(1 - countProgress, 3));
  const label = props.direction === 'buy' ? '推定取得金額' : '推定売却金額';

  return (
    <div style={{display: 'flex', flexDirection: 'column', gap: 26, alignItems: 'center'}}>
      <Stat label={label} value={shown.toFixed(1)} unit="億円" accent={accent} big />
      <Stat
        label="保有比率"
        value={props.holdingRatio.toFixed(2)}
        unit="%"
        accent={brand.blue}
      />
    </div>
  );
};

const FilerVisual: React.FC<{props: ShortProps; accent: string}> = ({props, accent}) => (
  <Card>
    <Label text="買い手は誰？" color={accent} />
    {props.filerName ? (
      <div
        style={{
          background: PLATE_BG,
          borderRadius: 24,
          padding: '26px 34px',
          fontSize: 56,
          fontWeight: 900,
          color: '#ffffff',
          lineHeight: 1.3,
        }}
      >
        {props.filerName}
      </div>
    ) : null}
    <div
      style={{
        marginTop: 18,
        display: 'inline-block',
        fontSize: 36,
        fontWeight: 900,
        color: brand.navyDeep,
        background: accent,
        borderRadius: 999,
        padding: '12px 34px',
      }}
    >
      {props.dealTypeLabel}
    </div>
  </Card>
);

/**
 * 前回→今回の保有比率を2本のバーで見せる。前回が取れない回は build_script.py 側で
 * シーンごと落とすため、ここに来る時点で prevHoldingRatio は必ずある。
 */
const ChangeVisual: React.FC<{props: ShortProps; accent: string}> = ({props, accent}) => {
  const frame = useCurrentFrame();
  const grow = interpolate(frame, [6, 40], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const ratio = props.holdingRatio;
  const prev = props.prevHoldingRatio ?? 0;
  const diff = ratio - prev;
  // バーの基準は「大きい方の1.2倍（下限10%）」。20%超の開示でも振り切れない
  const scaleMax = Math.max(10, Math.max(ratio, prev) * 1.2);
  const barWidth = contentWidth - 220;
  const widthFor = (r: number) => Math.max(24, (r / scaleMax) * barWidth);

  return (
    <Card>
      <Label text="保有比率の推移" color={accent} />
      <div
        style={{
          background: PLATE_BG,
          borderRadius: 24,
          padding: '28px 30px',
          display: 'flex',
          flexDirection: 'column',
          gap: 24,
        }}
      >
        <Bar label="前回" ratio={prev} width={widthFor(prev) * grow} color="rgba(255,255,255,0.30)" />
        <Bar label="今回" ratio={ratio} width={widthFor(ratio) * grow} color={accent} />
        <div style={{fontSize: 48, fontWeight: 900, color: accent, textShadow: TEXT_SHADOW}}>
          {diff >= 0 ? '+' : '−'}
          {Math.abs(diff).toFixed(2)}ポイント
        </div>
      </div>
    </Card>
  );
};

const Bar: React.FC<{label: string; ratio: number; width: number; color: string}> = ({
  label,
  ratio,
  width,
  color,
}) => (
  <div style={{display: 'flex', alignItems: 'center', gap: 18}}>
    <span style={{fontSize: 30, fontWeight: 800, color: brand.cream, opacity: 0.75, width: 74}}>
      {label}
    </span>
    <div style={{height: 48, width, borderRadius: 10, background: color}} />
    <span style={{fontSize: 46, fontWeight: 900, color: '#ffffff'}}>{ratio.toFixed(2)}%</span>
  </div>
);

/** 直近3ヶ月の株価推移。線が左から右へ伸びる描画で「動き」を作る。 */
const ChartVisual: React.FC<{scene: Scene; accent: string}> = ({scene, accent}) => {
  const frame = useCurrentFrame();
  const closes = scene.closes ?? [];
  if (closes.length < 2) {
    return null;
  }

  const w = contentWidth;
  const h = 540;
  const padX = 70;
  const padY = 70;
  const lo = Math.min(...closes);
  const hi = Math.max(...closes);
  const span = hi - lo || 1;
  const n = closes.length;

  // 線の伸びる速さ: 約2.5秒で全体を描き切り、残りの時間は完成形を見せる
  const drawn = interpolate(frame, [6, 81], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const visibleCount = Math.max(2, Math.ceil(n * drawn));

  const pointAt = (i: number): [number, number] => [
    padX + (i / (n - 1)) * (w - padX * 2),
    padY + (h - padY * 2) * (1 - (closes[i] - lo) / span),
  ];
  const points = Array.from({length: visibleCount}, (_, i) => pointAt(i));
  const path = points
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ');
  const [headX, headY] = points[points.length - 1];

  const latest = closes[closes.length - 1];
  const changePct = (latest / closes[0] - 1) * 100;
  const up = changePct >= 0;
  const lineColor = up ? brand.buy : brand.sell;
  const dates = scene.dates ?? [];
  // 開示日が3ヶ月レンジの外（closes は直近63営業日）の回は縦線を描かない
  const discIndex =
    scene.discIndex !== undefined && scene.discIndex >= 0 && scene.discIndex < n
      ? scene.discIndex
      : null;

  return (
    <Card>
      <Label text="株価の推移（直近3ヶ月）" color={accent} />
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        <rect x={0} y={0} width={w} height={h} rx={24} fill={PLATE_BG} />
        {/* 高値・安値の目盛り */}
        <line x1={padX} y1={padY} x2={w - padX} y2={padY} stroke="#ffffff" strokeWidth={1} opacity={0.18} />
        <line
          x1={padX}
          y1={h - padY}
          x2={w - padX}
          y2={h - padY}
          stroke="#ffffff"
          strokeWidth={1}
          opacity={0.18}
        />
        <text x={padX} y={padY - 14} fontSize={26} fill={brand.cream} opacity={0.7}>
          高値 {Math.round(hi).toLocaleString('ja-JP')}円
        </text>
        <text x={padX} y={h - padY + 36} fontSize={26} fill={brand.cream} opacity={0.7}>
          安値 {Math.round(lo).toLocaleString('ja-JP')}円
        </text>
        {/* 開示日の位置 */}
        {discIndex !== null ? (
          <>
            <line
              x1={pointAt(discIndex)[0]}
              y1={padY - 4}
              x2={pointAt(discIndex)[0]}
              y2={h - padY + 4}
              stroke={brand.goldBright}
              strokeWidth={4}
              strokeDasharray="12 10"
            />
            <text
              x={pointAt(discIndex)[0]}
              y={padY - 44}
              fontSize={28}
              fontWeight={800}
              fill={brand.goldBright}
              textAnchor="middle"
            >
              開示
            </text>
          </>
        ) : null}
        <path
          d={`${path} L${headX.toFixed(1)},${h - padY} L${padX},${h - padY} Z`}
          fill={lineColor}
          opacity={0.16}
        />
        <path d={path} stroke={lineColor} strokeWidth={10} fill="none" strokeLinecap="round" />
        <circle cx={headX} cy={headY} r={15} fill={lineColor} />
        {dates.length === n ? (
          <text
            x={w - padX}
            y={h - 18}
            fontSize={24}
            fill={brand.cream}
            opacity={0.6}
            textAnchor="end"
          >
            {dates[0]} 〜 {dates[n - 1]}
          </text>
        ) : null}
      </svg>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 30,
          alignItems: 'baseline',
          marginTop: 16,
        }}
      >
        <span style={{fontSize: 62, fontWeight: 900, color: '#ffffff', textShadow: TEXT_SHADOW}}>
          {latest.toLocaleString('ja-JP', {maximumFractionDigits: 0})}円
        </span>
        <span style={{fontSize: 52, fontWeight: 900, color: lineColor, textShadow: TEXT_SHADOW}}>
          {up ? '+' : ''}
          {changePct.toFixed(1)}%
        </span>
      </div>
    </Card>
  );
};

/**
 * 締め。URLの手打ちは期待せず「検索」に落とす。末尾0.8秒は冒頭と同じ金額組版に
 * カットで戻し、ループ再生で0秒目と絵が繋がるようにする（ループ視聴と相性が良い）。
 */
const CtaVisual: React.FC<{
  props: ShortProps;
  accent: string;
  durationInFrames: number;
}> = ({props, accent, durationInFrames}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 14, stiffness: 200}});

  if (frame >= durationInFrames - LOOP_TAIL_FRAMES) {
    return <HookVisual props={props} accent={accent} revealed />;
  }

  return (
    <div style={{textAlign: 'center', transform: `scale(${interpolate(enter, [0, 1], [0.9, 1])})`}}>
      {/* サイトの名乗りは kujira-watch/src/lib/site.ts の SITE_NAME に合わせる。
          「クジラウォッチ」という名前では運営していないので動画側で勝手に名乗らない */}
      <div style={{fontSize: 64, fontWeight: 900, color: '#ffffff', textShadow: TEXT_SHADOW}}>
        大口投資家の監視ブログ
      </div>
      <div
        style={{
          display: 'inline-block',
          marginTop: 22,
          fontSize: 44,
          fontWeight: 900,
          color: brand.navyDeep,
          background: brand.goldBright,
          borderRadius: 999,
          padding: '16px 44px',
        }}
      >
        kujira-watch.com
      </div>
      {/* VOICEVOX利用規約で必須のクレジット表記（音声が無い回も出て害はない） */}
      <div style={{fontSize: 22, color: brand.cream, opacity: 0.5, marginTop: 20}}>
        音声: VOICEVOX:ずんだもん
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ */

const Card: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{textAlign: 'center', maxWidth: contentWidth}}>{children}</div>
);

/**
 * 見出しラベル。以前は色文字だけで背景に溶けていたため、不透明の下地を敷き、
 * アクセント色は下線に逃がした（2026-08-19）。
 */
const Label: React.FC<{text: string; color: string}> = ({text, color}) => (
  <div
    style={{
      display: 'inline-block',
      fontSize: 34,
      letterSpacing: 4,
      color: '#ffffff',
      fontWeight: 900,
      background: 'rgba(6,10,20,0.78)',
      borderRadius: 999,
      borderBottom: `4px solid ${color}`,
      padding: '10px 26px',
      marginBottom: 20,
      textShadow: TEXT_SHADOW,
    }}
  >
    {text}
  </div>
);

const Stat: React.FC<{
  label: string;
  value: string;
  unit: string;
  accent: string;
  big?: boolean;
}> = ({label, value, unit, accent, big}) => (
  <div
    style={{
      background: PLATE_BG,
      border: `4px solid ${accent}`,
      borderRadius: 24,
      padding: big ? '30px 54px' : '22px 44px',
      textAlign: 'center',
    }}
  >
    <div style={{fontSize: 30, color: brand.cream, opacity: 0.8, marginBottom: 8}}>{label}</div>
    <div style={{display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 8}}>
      <span
        style={{
          fontSize: big ? 132 : 88,
          fontWeight: 900,
          color: '#ffffff',
          lineHeight: 1,
          textShadow: TEXT_SHADOW,
        }}
      >
        {value}
      </span>
      <span style={{fontSize: big ? 52 : 40, fontWeight: 900, color: accent}}>{unit}</span>
    </div>
  </div>
);

const formatAmount = (oku: number): string =>
  oku >= 100 ? Math.round(oku).toLocaleString() : oku.toFixed(1).replace(/\.0$/, '');

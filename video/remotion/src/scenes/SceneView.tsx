import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {accentFor, brand, fontFamily, safeArea} from '../theme';
import type {Scene, ShortProps} from '../types';

/**
 * 1シーンの見た目。kind ごとに中央のビジュアルを切り替え、下段に字幕を大きく出す。
 *
 * 設計方針（TikTok運用の定石に合わせる）:
 * - 無音で見る視聴者が多数派なので、ナレーションの要約字幕を常に大きく表示する
 * - 表示はすべて safeArea 内。下部はTikTokのキャプション・右端はボタン列に隠れるため
 * - カット感を出すため、シーン頭は spring で「飛び込ませる」。フェードは使わない
 *   （ゆっくりしたフェードはショート動画では「遅い」と感じられる）
 */
export const SceneView: React.FC<{
  scene: Scene;
  props: ShortProps;
  sceneIndex: number;
}> = ({scene, props, sceneIndex}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 16, stiffness: 180}});
  const accent = accentFor(props.direction);

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
      {/* 上段: 銘柄の常時表示（どのシーンから見始めても文脈がわかるように） */}
      <Ticker props={props} show={scene.kind !== 'hook' && scene.kind !== 'cta'} />

      {/* 中段: kindごとのビジュアル */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          transform: `translateY(${interpolate(enter, [0, 1], [56, 0])}px)`,
          opacity: interpolate(enter, [0, 0.35], [0, 1], {extrapolateRight: 'clamp'}),
        }}
      >
        <Visual scene={scene} props={props} accent={accent} />
      </div>

      {/* 下段: 字幕（無音視聴者のための主役。ナレーションの要約＋全文） */}
      <Caption
        text={scene.caption}
        narration={scene.kind === 'hook' || scene.kind === 'cta' ? '' : scene.narration}
        accent={accent}
        kind={scene.kind}
      />
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ */

const Ticker: React.FC<{props: ShortProps; show: boolean}> = ({props, show}) => {
  if (!show) {
    return <div style={{height: 64}} />;
  }
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 18,
        height: 64,
      }}
    >
      <span style={{fontSize: 40}}>🐋</span>
      <span style={{fontSize: 38, fontWeight: 900, color: '#ffffff'}}>
        {props.stockName}
      </span>
      <span style={{fontSize: 30, fontWeight: 700, color: brand.cream, opacity: 0.6}}>
        {props.stockCode}
      </span>
      <span
        style={{
          marginLeft: 'auto',
          fontSize: 28,
          fontWeight: 800,
          color: props.direction === 'buy' ? brand.buy : brand.sell,
          background: 'rgba(255,255,255,0.08)',
          borderRadius: 999,
          padding: '8px 22px',
        }}
      >
        {props.direction === 'buy' ? '買い' : '売り'}
      </span>
    </div>
  );
};

const Caption: React.FC<{
  text: string;
  narration: string;
  accent: string;
  kind: Scene['kind'];
}> = ({text, narration, accent, kind}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  // 字幕は本文より一拍遅れて入る（視線が中央→下段の順で動くように）
  const enter = spring({frame: frame - 4, fps, config: {damping: 15, stiffness: 200}});
  if (!text) {
    return null;
  }
  const isSpeculation = kind === 'outlook';
  return (
    <div
      style={{
        alignSelf: 'center',
        transform: `scale(${interpolate(enter, [0, 1], [0.92, 1])})`,
        opacity: interpolate(enter, [0, 0.4], [0, 1], {extrapolateRight: 'clamp'}),
        background: 'rgba(6,10,20,0.82)',
        border: `3px solid ${isSpeculation ? brand.blue : accent}`,
        borderRadius: 20,
        padding: '26px 38px',
        maxWidth: 820,
      }}
    >
      {isSpeculation ? (
        <div style={{fontSize: 26, fontWeight: 800, color: brand.blue, marginBottom: 8}}>
          ※ここから先は推測
        </div>
      ) : null}
      <div
        style={{
          fontSize: 54,
          fontWeight: 900,
          lineHeight: 1.35,
          color: '#ffffff',
          textAlign: 'center',
        }}
      >
        {text}
      </div>
      {/* 無音視聴者向けのナレーション全文。シーンが10秒前後あるため、
          26字の要約だけでは間が持たない */}
      {narration ? (
        <div
          style={{
            fontSize: 34,
            fontWeight: 600,
            lineHeight: 1.6,
            color: brand.cream,
            opacity: 0.85,
            textAlign: 'center',
            marginTop: 16,
          }}
        >
          {narration}
        </div>
      ) : null}
    </div>
  );
};

/* ------------------------------------------------------------------ */

const Visual: React.FC<{scene: Scene; props: ShortProps; accent: string}> = ({
  scene,
  props,
  accent,
}) => {
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
    case 'outlook':
      // 中央ビジュアル無し。※推測ラベル付きの字幕カードだけで語らせる
      return null;
    case 'chart':
      return <ChartVisual scene={scene} props={props} accent={accent} />;
    case 'cta':
      return <CtaVisual />;
    default:
      return null;
  }
};

/**
 * 冒頭0.5秒で金額を画面いっぱいに叩き込む。ブランドや日付の前置きは一切しない
 * （最初の1秒で離脱が決まるため、一番強い数字から入る）。
 */
const HookVisual: React.FC<{props: ShortProps; accent: string}> = ({props, accent}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const slam = spring({frame, fps, config: {damping: 11, stiffness: 240}});
  // 言い切りの動詞だけで締める。決め台詞的な文言は付けない
  // （興味の引きはClaudeが記事から作る字幕captionに任せる）。
  const verb = props.direction === 'buy' ? '買い集めた' : '売り払った';

  return (
    <div style={{textAlign: 'center'}}>
      <div
        style={{
          fontSize: 64,
          fontWeight: 900,
          color: '#ffffff',
          opacity: interpolate(frame, [6, 12], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
        }}
      >
        {props.stockName}を
      </div>
      <div
        style={{
          fontSize: 210,
          fontWeight: 900,
          lineHeight: 1.05,
          color: accent,
          transform: `scale(${interpolate(slam, [0, 1], [1.6, 1])})`,
          textShadow: '0 10px 60px rgba(0,0,0,0.6)',
        }}
      >
        {formatAmount(props.dealAmountOku)}
        <span style={{fontSize: 100}}>億円</span>
      </div>
      <div
        style={{
          fontSize: 72,
          fontWeight: 900,
          color: '#ffffff',
          opacity: interpolate(frame, [10, 16], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
        }}
      >
        {verb}
      </div>
    </div>
  );
};

const CompanyVisual: React.FC<{props: ShortProps}> = ({props}) => (
  <Card>
    <Label text="どんな会社？" color={brand.goldBright} />
    <div style={{fontSize: 88, fontWeight: 900, color: '#ffffff', lineHeight: 1.2}}>
      {props.stockName}
    </div>
    <div style={{fontSize: 42, color: brand.cream, opacity: 0.65, marginTop: 8}}>
      証券コード {props.stockCode}
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
      <div style={{fontSize: 58, fontWeight: 900, color: '#ffffff', lineHeight: 1.3}}>
        {props.filerName}
      </div>
    ) : null}
    <div
      style={{
        marginTop: 18,
        display: 'inline-block',
        fontSize: 36,
        fontWeight: 800,
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

/** 前回→今回の保有比率をバーの伸縮で見せる。 */
const ChangeVisual: React.FC<{props: ShortProps; accent: string}> = ({props, accent}) => {
  const frame = useCurrentFrame();
  const grow = interpolate(frame, [6, 40], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const ratio = props.holdingRatio;
  // バーの最大幅を保有比率10%相当とする（大量保有の実務レンジでバーの差が見えるように）
  const widthFor = (r: number) => Math.min(720, Math.max(60, (r / 10) * 720));

  return (
    <Card>
      <Label text="保有比率の推移" color={accent} />
      <div style={{display: 'flex', flexDirection: 'column', gap: 24, marginTop: 12}}>
        <div>
          <div style={{fontSize: 32, color: brand.cream, opacity: 0.7, marginBottom: 8}}>
            今回の開示
          </div>
          <div style={{display: 'flex', alignItems: 'center', gap: 20}}>
            <div
              style={{
                height: 56,
                width: widthFor(ratio) * grow,
                borderRadius: 12,
                background: accent,
              }}
            />
            <span style={{fontSize: 60, fontWeight: 900, color: '#ffffff'}}>
              {ratio.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
};

/** 直近3ヶ月の株価推移。線が左から右へ伸びる描画で「動き」を作る。 */
const ChartVisual: React.FC<{scene: Scene; props: ShortProps; accent: string}> = ({
  scene,
  props,
  accent,
}) => {
  const frame = useCurrentFrame();
  const closes = scene.closes ?? [];
  if (closes.length < 2) {
    return null;
  }

  const w = 800;
  const h = 560;
  const padX = 20;
  const padY = 60;
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
  const path = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const [headX, headY] = points[points.length - 1];

  const latest = closes[closes.length - 1];
  const changePct = (latest / closes[0] - 1) * 100;
  const up = changePct >= 0;
  const lineColor = up ? brand.buy : brand.sell;

  return (
    <Card>
      <Label text="株価の推移（直近3ヶ月）" color={accent} />
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{overflow: 'visible'}}>
        {/* 面グラデーション（描画済みの範囲だけ） */}
        <path
          d={`${path} L${headX.toFixed(1)},${h - padY} L${padX},${h - padY} Z`}
          fill={lineColor}
          opacity={0.14}
        />
        <path d={path} stroke={lineColor} strokeWidth={7} fill="none" strokeLinecap="round" />
        {/* 先端マーカー */}
        <circle cx={headX} cy={headY} r={13} fill={lineColor} />
      </svg>
      <div style={{display: 'flex', justifyContent: 'center', gap: 30, alignItems: 'baseline'}}>
        <span style={{fontSize: 62, fontWeight: 900, color: '#ffffff'}}>
          {latest.toLocaleString('ja-JP', {maximumFractionDigits: 0})}円
        </span>
        <span style={{fontSize: 52, fontWeight: 900, color: lineColor}}>
          {up ? '+' : ''}
          {changePct.toFixed(1)}%
        </span>
      </div>
    </Card>
  );
};

/**
 * 締め。冒頭と同じ「金額の一撃」で終わることで、ループ再生時に頭と自然に繋がる
 * （TikTokはループ回数も評価に入る）。免責は小さくsafeArea内に収める。
 */
const CtaVisual: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 14, stiffness: 200}});
  return (
    <div style={{textAlign: 'center', transform: `scale(${interpolate(enter, [0, 1], [0.9, 1])})`}}>
      <div style={{fontSize: 110}}>🐋</div>
      <div style={{fontSize: 74, fontWeight: 900, color: '#ffffff', marginTop: 8}}>
        クジラウォッチ
      </div>
      <div
        style={{
          display: 'inline-block',
          marginTop: 22,
          fontSize: 44,
          fontWeight: 800,
          color: brand.navyDeep,
          background: brand.goldBright,
          borderRadius: 999,
          padding: '16px 44px',
        }}
      >
        kujira-watch.com
      </div>
      <div style={{fontSize: 24, color: brand.cream, opacity: 0.55, marginTop: 24}}>
        ※公開情報の要約です。投資勧誘・投資助言ではありません
      </div>
      {/* VOICEVOX利用規約で必須のクレジット表記（音声が無い回も出て害はない） */}
      <div style={{fontSize: 22, color: brand.cream, opacity: 0.45, marginTop: 8}}>
        音声: VOICEVOX:ずんだもん
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ */

const Card: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{textAlign: 'center', maxWidth: 820}}>{children}</div>
);

const Label: React.FC<{text: string; color: string}> = ({text, color}) => (
  <div style={{fontSize: 34, letterSpacing: 6, color, fontWeight: 800, marginBottom: 20}}>
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
      background: 'rgba(255,255,255,0.07)',
      border: `3px solid ${accent}`,
      borderRadius: 24,
      padding: big ? '30px 54px' : '22px 44px',
      textAlign: 'center',
    }}
  >
    <div style={{fontSize: 30, color: brand.cream, opacity: 0.75, marginBottom: 8}}>{label}</div>
    <div style={{display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 8}}>
      <span
        style={{fontSize: big ? 132 : 88, fontWeight: 900, color: '#ffffff', lineHeight: 1}}
      >
        {value}
      </span>
      <span style={{fontSize: big ? 52 : 40, fontWeight: 800, color: accent}}>{unit}</span>
    </div>
  </div>
);

const formatAmount = (oku: number): string =>
  oku >= 100 ? Math.round(oku).toLocaleString() : oku.toFixed(1).replace(/\.0$/, '');

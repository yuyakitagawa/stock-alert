import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {brand, fontFamily} from '../theme';
import type {ShortProps} from '../types';

/**
 * 銘柄・提出者と、金額/保有比率という「数字」を見せるシーン。
 * 金額はカウントアップさせて、静止テキストより視線が留まるようにする。
 */
export const FactScene: React.FC<
  Pick<
    ShortProps,
    'stockName' | 'stockCode' | 'filerName' | 'dealTypeLabel' | 'direction' | 'dealAmountOku' | 'holdingRatio'
  >
> = ({stockName, stockCode, filerName, dealTypeLabel, direction, dealAmountOku, holdingRatio}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const accent = direction === 'buy' ? brand.goldBright : brand.sell;
  const label = direction === 'buy' ? '推定取得金額' : '推定売却金額';

  const countProgress = interpolate(frame, [10, 55], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const shownAmount = dealAmountOku * easeOutCubic(countProgress);

  return (
    <AbsoluteFill
      style={{
        fontFamily,
        justifyContent: 'center',
        alignItems: 'center',
        padding: '0 80px',
        textAlign: 'center',
        gap: 56,
      }}
    >
      <Reveal frame={frame} fps={fps} delay={0}>
        <div style={{fontSize: 32, letterSpacing: 4, color: accent, fontWeight: 800}}>
          {dealTypeLabel}
        </div>
        {/* filerName は古い記事だと未設定のことがあるため、空なら行ごと出さない
            （空の div を残すと分類ラベルと銘柄名の間が不自然に空く） */}
        {filerName ? (
          <div style={{fontSize: 44, color: brand.cream, opacity: 0.9, marginTop: 12}}>
            {filerName}
          </div>
        ) : null}
      </Reveal>

      <Reveal frame={frame} fps={fps} delay={8}>
        <div
          style={{
            fontSize: 76,
            fontWeight: 900,
            color: '#ffffff',
            lineHeight: 1.25,
          }}
        >
          {stockName}
        </div>
        <div style={{fontSize: 40, color: brand.cream, opacity: 0.65, marginTop: 8}}>
          ({stockCode})
        </div>
      </Reveal>

      <Reveal frame={frame} fps={fps} delay={16}>
        <div
          style={{
            display: 'flex',
            gap: 28,
            alignItems: 'stretch',
            justifyContent: 'center',
            marginTop: 24,
          }}
        >
          <Stat
            label={label}
            value={`${shownAmount.toFixed(1)}`}
            unit="億円"
            accent={accent}
          />
          <Stat
            label="保有比率"
            value={holdingRatio.toFixed(2)}
            unit="%"
            accent={brand.blue}
          />
        </div>
      </Reveal>
    </AbsoluteFill>
  );
};

const Stat: React.FC<{label: string; value: string; unit: string; accent: string}> = ({
  label,
  value,
  unit,
  accent,
}) => (
  <div
    style={{
      background: 'rgba(255,255,255,0.07)',
      border: `2px solid ${accent}`,
      borderRadius: 24,
      padding: '30px 38px',
      minWidth: 340,
    }}
  >
    <div style={{fontSize: 30, color: brand.cream, opacity: 0.75, marginBottom: 10}}>{label}</div>
    <div style={{display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 8}}>
      <span style={{fontSize: 96, fontWeight: 900, color: '#ffffff', lineHeight: 1}}>{value}</span>
      <span style={{fontSize: 40, fontWeight: 800, color: accent}}>{unit}</span>
    </div>
  </div>
);

const Reveal: React.FC<{
  frame: number;
  fps: number;
  delay: number;
  children: React.ReactNode;
}> = ({frame, fps, delay, children}) => {
  const progress = spring({frame: frame - delay, fps, config: {damping: 200}});
  return (
    <div
      style={{
        opacity: progress,
        transform: `translateY(${interpolate(progress, [0, 1], [30, 0])}px)`,
      }}
    >
      {children}
    </div>
  );
};

const easeOutCubic = (t: number): number => 1 - Math.pow(1 - t, 3);

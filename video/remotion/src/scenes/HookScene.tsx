import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {brand, fontFamily} from '../theme';
import type {ShortProps} from '../types';

/** 冒頭3〜4秒。スクロールを止めさせる「つかみ」だけを大きく出す。 */
export const HookScene: React.FC<Pick<ShortProps, 'hook' | 'direction' | 'discDate'>> = ({
  hook,
  direction,
  discDate,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 200}});
  const accent = direction === 'buy' ? brand.goldBright : brand.sell;

  return (
    <AbsoluteFill
      style={{
        fontFamily,
        justifyContent: 'center',
        alignItems: 'center',
        padding: '0 90px',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          opacity: interpolate(frame, [0, 12], [0, 1], {extrapolateRight: 'clamp'}),
          transform: `translateY(${interpolate(enter, [0, 1], [40, 0])}px)`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 44,
        }}
      >
        <div
          style={{
            fontSize: 34,
            letterSpacing: 6,
            color: brand.cream,
            opacity: 0.72,
            fontWeight: 700,
          }}
        >
          {formatDate(discDate)} の開示
        </div>
        <div style={{fontSize: 150, lineHeight: 1}}>{direction === 'buy' ? '🐋' : '📉'}</div>
        <div
          style={{
            fontSize: 86,
            fontWeight: 900,
            color: '#ffffff',
            lineHeight: 1.35,
            textShadow: '0 8px 40px rgba(0,0,0,0.45)',
          }}
        >
          {hook}
        </div>
        <div
          style={{
            height: 8,
            width: interpolate(enter, [0, 1], [0, 260]),
            borderRadius: 4,
            background: accent,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};

const formatDate = (iso: string): string => {
  const [, m, d] = iso.split('-');
  return m && d ? `${Number(m)}月${Number(d)}日` : iso;
};

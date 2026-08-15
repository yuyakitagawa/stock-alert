import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {brand, fontFamily} from '../theme';
import type {ShortProps} from '../types';

/** 記事の要点を3行で順に出す。1行あたり約2秒で読み切れる分量を想定。 */
export const BulletScene: React.FC<Pick<ShortProps, 'bullets' | 'direction'>> = ({
  bullets,
  direction,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const accent = direction === 'buy' ? brand.goldBright : brand.sell;
  // build_script.py 側で40字に制限しているが、長めの行が来たときに3行へ折り返して
  // 画面からはみ出さないよう、文字量に応じて1段階だけ縮める。
  const longest = bullets.reduce((max, b) => Math.max(max, b.length), 0);
  const bulletFontSize = longest > 30 ? 46 : 52;

  return (
    <AbsoluteFill
      style={{
        fontFamily,
        justifyContent: 'center',
        padding: '0 90px',
        gap: 52,
      }}
    >
      <div style={{fontSize: 36, letterSpacing: 6, color: accent, fontWeight: 800}}>
        POINT
      </div>
      {bullets.map((text, i) => {
        const progress = spring({frame: frame - i * 26, fps, config: {damping: 200}});
        return (
          <div
            key={i}
            style={{
              display: 'flex',
              gap: 28,
              alignItems: 'flex-start',
              opacity: progress,
              transform: `translateX(${interpolate(progress, [0, 1], [-50, 0])}px)`,
            }}
          >
            <div
              style={{
                flexShrink: 0,
                width: 62,
                height: 62,
                borderRadius: 31,
                background: accent,
                color: brand.navyDeep,
                fontSize: 34,
                fontWeight: 900,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {i + 1}
            </div>
            <div
              style={{
                fontSize: bulletFontSize,
                fontWeight: 700,
                color: '#ffffff',
                lineHeight: 1.45,
              }}
            >
              {text}
            </div>
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

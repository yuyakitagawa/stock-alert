import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {brand, fontFamily} from '../theme';
import type {ShortProps} from '../types';

/** 締め。サイト名とドメインを出して kujira-watch.com への流入につなげる。 */
export const CtaScene: React.FC<Pick<ShortProps, 'closing'>> = ({closing}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 200}});

  return (
    <AbsoluteFill
      style={{
        fontFamily,
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        padding: '0 90px',
        gap: 40,
      }}
    >
      <div
        style={{
          fontSize: 58,
          fontWeight: 700,
          color: brand.cream,
          opacity: interpolate(frame, [0, 12], [0, 1], {extrapolateRight: 'clamp'}),
        }}
      >
        {closing}
      </div>
      <div
        style={{
          transform: `scale(${interpolate(enter, [0, 1], [0.86, 1])})`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 16,
        }}
      >
        <div style={{fontSize: 120}}>🐋</div>
        <div style={{fontSize: 78, fontWeight: 900, color: '#ffffff'}}>クジラウォッチ</div>
        <div
          style={{
            fontSize: 46,
            fontWeight: 800,
            color: brand.navyDeep,
            background: brand.goldBright,
            borderRadius: 999,
            padding: '18px 46px',
          }}
        >
          kujira-watch.com
        </div>
      </div>
      <div style={{fontSize: 28, color: brand.cream, opacity: 0.6, marginTop: 20, lineHeight: 1.6}}>
        ※本動画は公開情報の要約であり、投資勧誘・投資助言ではありません
      </div>
    </AbsoluteFill>
  );
};

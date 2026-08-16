import React from 'react';
import {useCurrentFrame} from 'remotion';
import {brand, safeArea} from './theme';

/**
 * 残り時間がわかる進行バー。「あと少しで終わる」が見えると完走率が上がる。
 * TikTok自体のシークバーは再生中は非表示のため、動画側に描く価値がある。
 */
export const ProgressBar: React.FC<{totalFrames: number}> = ({totalFrames}) => {
  const frame = useCurrentFrame();
  const progress = Math.min(1, frame / Math.max(1, totalFrames));

  return (
    <div
      style={{
        position: 'absolute',
        top: safeArea.top - 60,
        left: safeArea.left,
        right: safeArea.right,
        height: 10,
        borderRadius: 5,
        background: 'rgba(255,255,255,0.15)',
      }}
    >
      <div
        style={{
          width: `${progress * 100}%`,
          height: '100%',
          borderRadius: 5,
          background: brand.goldBright,
        }}
      />
    </div>
  );
};

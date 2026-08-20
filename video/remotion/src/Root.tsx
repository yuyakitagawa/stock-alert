import React from 'react';
import {Composition} from 'remotion';
import {ArticleShort} from './ArticleShort';
import {defaultShortProps, sceneDurationSec, type ShortProps} from './types';

const FPS = 30;

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="ArticleShort"
      component={ArticleShort}
      fps={FPS}
      // YouTube Shorts の 9:16。1080x1920 が推奨解像度。
      width={1080}
      height={1920}
      defaultProps={defaultShortProps}
      // 尺は固定ではなく、propsのシーン数とナレーションの長さで決まる
      // （ArticleShort.tsx のシーケンス計算と同じ式）。
      durationInFrames={30 * 30}
      calculateMetadata={({props}) => {
        const p = props as ShortProps;
        const totalSec = p.scenes.reduce((sum, s) => sum + sceneDurationSec(s), 0);
        return {durationInFrames: Math.max(FPS * 5, Math.round(totalSec * FPS))};
      }}
    />
  );
};

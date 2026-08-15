import React from 'react';
import {Composition} from 'remotion';
import {ArticleShort, TOTAL_FRAMES} from './ArticleShort';
import {defaultShortProps} from './types';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="ArticleShort"
      component={ArticleShort}
      durationInFrames={TOTAL_FRAMES}
      fps={30}
      // YouTube Shorts / TikTok 共通の 9:16。1080x1920 が両方の推奨解像度。
      width={1080}
      height={1920}
      defaultProps={defaultShortProps}
    />
  );
};

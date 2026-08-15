import React from 'react';
import {AbsoluteFill, Sequence} from 'remotion';
import {Background} from './Background';
import {HookScene} from './scenes/HookScene';
import {FactScene} from './scenes/FactScene';
import {BulletScene} from './scenes/BulletScene';
import {CtaScene} from './scenes/CtaScene';
import type {ShortProps} from './types';

/**
 * シーンの尺（frame数, 30fps）。合計 600 frames = 20秒。
 * YouTube Shorts / TikTok のどちらも、最後まで見てもらいやすい 15〜25秒に収める。
 */
export const SCENE_FRAMES = {
  hook: 110,
  fact: 190,
  bullets: 200,
  cta: 100,
} as const;

export const TOTAL_FRAMES =
  SCENE_FRAMES.hook + SCENE_FRAMES.fact + SCENE_FRAMES.bullets + SCENE_FRAMES.cta;

export const ArticleShort: React.FC<ShortProps> = (props) => {
  const hookStart = 0;
  const factStart = hookStart + SCENE_FRAMES.hook;
  const bulletsStart = factStart + SCENE_FRAMES.fact;
  const ctaStart = bulletsStart + SCENE_FRAMES.bullets;

  return (
    <AbsoluteFill>
      <Background />
      <Sequence from={hookStart} durationInFrames={SCENE_FRAMES.hook}>
        <HookScene hook={props.hook} direction={props.direction} discDate={props.discDate} />
      </Sequence>
      <Sequence from={factStart} durationInFrames={SCENE_FRAMES.fact}>
        <FactScene
          stockName={props.stockName}
          stockCode={props.stockCode}
          filerName={props.filerName}
          dealTypeLabel={props.dealTypeLabel}
          direction={props.direction}
          dealAmountOku={props.dealAmountOku}
          holdingRatio={props.holdingRatio}
        />
      </Sequence>
      <Sequence from={bulletsStart} durationInFrames={SCENE_FRAMES.bullets}>
        <BulletScene bullets={props.bullets} direction={props.direction} />
      </Sequence>
      <Sequence from={ctaStart} durationInFrames={SCENE_FRAMES.cta}>
        <CtaScene closing={props.closing} />
      </Sequence>
    </AbsoluteFill>
  );
};

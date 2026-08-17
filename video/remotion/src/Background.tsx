import React from 'react';
import {
  AbsoluteFill,
  Loop,
  OffthreadVideo,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {brand} from './theme';

/**
 * 全シーン共通の背景。Pexelsの自然映像（海系・縦向き）があればそれをループ再生し、
 * 文字が読めるよう暗いオーバーレイを重ねる。映像が無い日は従来のグラデーション＋波に
 * フォールバックする。いずれも「1フレームも完全静止させない」ことで、静止画
 * スライドショーに見えるのを防ぐ（TikTokでは画面が止まった瞬間にスワイプされる）。
 */
export const Background: React.FC<{
  videoFile?: string;
  videoDurationSec?: number;
}> = ({videoFile, videoDurationSec}) => {
  const {fps} = useVideoConfig();

  if (videoFile) {
    // 素材の尺ぶんでループ。端数フレームの黒落ちを避けるため floor で切る。
    const loopFrames = Math.max(fps, Math.floor((videoDurationSec ?? 10) * fps));
    return (
      <AbsoluteFill style={{overflow: 'hidden', background: brand.navyDeep}}>
        <Loop durationInFrames={loopFrames}>
          <OffthreadVideo
            src={staticFile(videoFile)}
            muted
            style={{width: '100%', height: '100%', objectFit: 'cover'}}
          />
        </Loop>
        {/* 文字の可読性を確保する薄めの暗幕。「黒すぎる」とのオーナー指摘（2026-08-17）で
            大幅に薄くした。可読性は字幕カード自体の濃い背景と本文のシャドウで担保する */}
        <AbsoluteFill
          style={{
            background: `linear-gradient(180deg, rgba(13,21,38,0.45) 0%, rgba(13,21,38,0.22) 40%, rgba(13,21,38,0.28) 70%, rgba(13,21,38,0.55) 100%)`,
          }}
        />
      </AbsoluteFill>
    );
  }

  return <GradientBackground />;
};

const GradientBackground: React.FC = () => {
  const frame = useCurrentFrame();
  // 20〜60秒かけてわずかに寄る。気づかれない程度のドリフトが「映像感」を作る。
  const zoom = 1 + frame * 0.00012;

  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      <AbsoluteFill
        style={{
          transform: `scale(${zoom})`,
          background: `radial-gradient(120% 80% at 50% 0%, ${brand.blueDark} 0%, ${brand.navy} 45%, ${brand.navyDeep} 100%)`,
        }}
      >
        <Wave y={1500} speed={1.6} opacity={0.18} color={brand.blue} amplitude={44} />
        <Wave y={1600} speed={-1.1} opacity={0.12} color={brand.goldBright} amplitude={60} />
        <Grain />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const Wave: React.FC<{
  y: number;
  speed: number;
  opacity: number;
  color: string;
  amplitude: number;
}> = ({y, speed, opacity, color, amplitude}) => {
  const frame = useCurrentFrame();
  const shift = frame * speed;
  const period = 540;
  const path = buildWavePath(y, amplitude, period, shift);

  return (
    <svg
      style={{position: 'absolute', inset: 0}}
      width={1080}
      height={1920}
      viewBox="0 0 1080 1920"
    >
      <path d={path} fill={color} opacity={opacity} />
    </svg>
  );
};

const buildWavePath = (
  y: number,
  amplitude: number,
  period: number,
  shift: number,
): string => {
  const points: string[] = [];
  for (let x = 0; x <= 1080; x += 20) {
    const wave = Math.sin(((x + shift) / period) * Math.PI * 2) * amplitude;
    points.push(`${x},${(y + wave).toFixed(1)}`);
  }
  return `M0,1920 L0,${y} L${points.join(' L')} L1080,1920 Z`;
};

/** ベタ塗りのグラデーションに出るバンディングを目立たなくするための微細なノイズ。 */
const Grain: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame % 6, [0, 3, 6], [0.035, 0.05, 0.035]);
  return (
    <AbsoluteFill
      style={{
        opacity,
        backgroundImage:
          'radial-gradient(#ffffff 1px, transparent 1px), radial-gradient(#ffffff 1px, transparent 1px)',
        backgroundSize: '7px 7px, 11px 11px',
        backgroundPosition: '0 0, 3px 5px',
        mixBlendMode: 'overlay',
      }}
    />
  );
};

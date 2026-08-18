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
 * 背景。Pexelsの自然映像がある区間（company / filer の2シーンのみ）はそれをループ再生し、
 * それ以外の「数字を読ませるシーン」はブランドのグラデーション背景に固定する
 * （実写の明部に数字が沈む事故を構造的に無くすため。2026-08-19）。
 * いずれも「1フレームも完全静止させない」ことで、静止画スライドショーに見えるのを防ぐ。
 */
export const Background: React.FC<{
  videoFile?: string;
  videoDurationSec?: number;
}> = ({videoFile, videoDurationSec}) => {
  const {fps} = useVideoConfig();
  const frame = useCurrentFrame();

  if (videoFile) {
    // 素材の尺ぶんでループ。端数フレームの黒落ちを避けるため floor で切る。
    const loopFrames = Math.max(fps, Math.floor((videoDurationSec ?? 10) * fps));
    return (
      <AbsoluteFill style={{overflow: 'hidden', background: brand.navyDeep}}>
        {/* Ken Burns。シーン内フレームで進むので、シーンが変わるたび等倍に戻る */}
        <AbsoluteFill style={{transform: `scale(${1 + frame * 0.00035})`}}>
          <Loop durationInFrames={loopFrames}>
            <OffthreadVideo
              src={staticFile(videoFile)}
              muted
              style={{width: '100%', height: '100%', objectFit: 'cover'}}
            />
          </Loop>
        </AbsoluteFill>
        {/* 実写の上に文字を置く2シーンにだけ効く暗幕。数字シーンはグラデ背景なので
            「黒すぎる」というオーナー指摘（2026-08-17）の対象にはならない */}
        <AbsoluteFill
          style={{
            background: `linear-gradient(180deg, rgba(13,21,38,0.62) 0%, rgba(13,21,38,0.45) 40%, rgba(13,21,38,0.50) 70%, rgba(13,21,38,0.72) 100%)`,
          }}
        />
      </AbsoluteFill>
    );
  }

  return <GradientBackground />;
};

const GradientBackground: React.FC = () => {
  const frame = useCurrentFrame();
  // シーンの尺ぶん（数秒）でわずかに寄る。気づかれない程度のドリフトが「映像感」を作る。
  const zoom = 1 + frame * 0.0003;

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

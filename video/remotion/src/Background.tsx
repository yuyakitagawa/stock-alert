import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {brand} from './theme';

/**
 * 全シーン共通の背景。クジラ＝海のイメージに合わせた濃紺のグラデーションに、
 * ゆっくり流れる波を2枚重ねる。動画全体を通して動きが途切れないようにすることで、
 * 静止画スライドショーに見えるのを防ぐ。
 */
export const Background: React.FC = () => {
  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(120% 80% at 50% 0%, ${brand.blueDark} 0%, ${brand.navy} 45%, ${brand.navyDeep} 100%)`,
      }}
    >
      <Wave y={1420} speed={0.55} opacity={0.16} color={brand.blue} amplitude={38} />
      <Wave y={1520} speed={-0.38} opacity={0.1} color={brand.goldBright} amplitude={52} />
      <Grain />
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
  // 波1周期ぶん余分に描いて左右に動かすことで、継ぎ目なくループしているように見せる。
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

import React from 'react';
import {AbsoluteFill, Audio, Sequence, staticFile, useVideoConfig} from 'remotion';
import {Background} from './Background';
import {SceneView} from './scenes/SceneView';
import {ProgressBar} from './ProgressBar';
import {sceneDurationSec, type ShortProps} from './types';

/**
 * シーンの尺は固定ではなく、各シーンのナレーション音声の長さ（音声が無い場合は
 * 読み上げ文字数からの見積もり）で決まる。総フレーム数は Root.tsx の
 * calculateMetadata が同じ計算で算出する。
 */
export const ArticleShort: React.FC<ShortProps> = (props) => {
  const {fps} = useVideoConfig();

  let cursor = 0;
  const sequences = props.scenes.map((scene, i) => {
    const durationInFrames = Math.round(sceneDurationSec(scene) * fps);
    const from = cursor;
    cursor += durationInFrames;
    return {scene, from, durationInFrames, key: `${scene.kind}-${i}`};
  });

  return (
    <AbsoluteFill>
      <Background />
      {sequences.map(({scene, from, durationInFrames, key}, i) => (
        <Sequence key={key} from={from} durationInFrames={durationInFrames}>
          {scene.audio ? <Audio src={staticFile(scene.audio)} /> : null}
          <SceneView scene={scene} props={props} sceneIndex={i} />
        </Sequence>
      ))}
      <ProgressBar totalFrames={cursor} />
    </AbsoluteFill>
  );
};

import React from 'react';
import {AbsoluteFill, Audio, Sequence, staticFile, useVideoConfig} from 'remotion';
import {Background} from './Background';
import {SceneView} from './scenes/SceneView';
import {sceneDurationSec, type ShortProps} from './types';

/**
 * シーンの尺は固定ではなく、各シーンのナレーション音声の長さ（音声が無い場合は
 * 読み上げ文字数からの見積もり）で決まる。総フレーム数は Root.tsx の
 * calculateMetadata が同じ計算で算出する。
 *
 * 効果音（props.sfx）は render.py が numpy で生成して public/ に置いたときだけ鳴らす。
 * 素材を外部から取ってこないので、毎日の全自動運用でライセンス確認が要らない。
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
      {/* 数字を読ませるシーンはブランドのグラデーション背景。company / filer だけ
          Pexelsの実写がシーン別に上書きされる */}
      <Background />
      {sequences.map(({scene, from, durationInFrames, key}, i) => (
        <Sequence key={key} from={from} durationInFrames={durationInFrames}>
          {scene.backgroundVideo ? (
            <Background
              videoFile={scene.backgroundVideo}
              videoDurationSec={scene.backgroundVideoDurationSec}
            />
          ) : null}
          {scene.audio ? <Audio src={staticFile(scene.audio)} /> : null}
          {/* カット頭のホワイトノイズ掃引。無音の切れ目が「再生バグ」に聞こえるのを防ぐ */}
          {props.sfx ? <Audio src={staticFile('se_whoosh.wav')} volume={0.26} /> : null}
          {/* 金額の着地と、金額カウントアップの終わりに1発ずつ */}
          {props.sfx && scene.kind === 'hook' ? (
            <Sequence from={8} durationInFrames={30}>
              <Audio src={staticFile('se_impact.wav')} volume={0.32} />
            </Sequence>
          ) : null}
          {props.sfx && scene.kind === 'deal' ? (
            <Sequence from={34} durationInFrames={15}>
              <Audio src={staticFile('se_tick.wav')} volume={0.22} />
            </Sequence>
          ) : null}
          <SceneView
            scene={scene}
            props={props}
            sceneIndex={i}
            durationInFrames={durationInFrames}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

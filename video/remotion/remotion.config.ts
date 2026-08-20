import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
// YouTube Shorts は H.264 + AAC の mp4 を推奨している。
Config.setCodec('h264');
// CRF 23: 実写背景込みでもファイルサイズが現実的な範囲（〜60MB/90秒）に収まる値。
// 当初は文字のにじみ対策でCRF 18にしていたが、実写の自然映像を背景に敷いた時点で
// 190MB/90秒まで膨らんだため引き上げた。YouTubeは投稿後に再圧縮するので、
// この差は最終画質にほぼ影響しない。
Config.setCrf(23);

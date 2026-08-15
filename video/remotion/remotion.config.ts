import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
// YouTube Shorts / TikTok はどちらも H.264 + AAC の mp4 を推奨している。
Config.setCodec('h264');
// 文字が主役の動画なので、圧縮によるテキストのにじみを避けるため CRF を低めにする。
Config.setCrf(18);

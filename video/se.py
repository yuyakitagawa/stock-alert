"""
video/se.py

効果音（SE）を numpy で自前生成する。

カット頭の無音が「演出ではなく再生バグに聞こえる」という指摘が
2026-08-19のインフルエンサーレビューで最多だったため、シーンの切り替わりと
金額の着地に短いSEを置く。フリー素材を毎日ダウンロードするとライセンス確認が
自動化できないので、波形をその場で合成する（著作権が発生しないので全自動運用と両立する）。

- se_whoosh.wav: カット頭。ローパスを開きながらのノイズスイープ（0.25秒）
- se_impact.wav: 金額の着地。低域のドン（0.30秒）
- se_tick.wav  : 数字のカウントアップ完了。短いチック（0.04秒）

numpy が無い環境では False を返し、呼び出し側はSE無しで動画を書き出す。
"""
import math
import os
import wave

SAMPLE_RATE = 44100

# render.py が public/ へ書き出し、レンダリング後に消すファイル名。
SE_FILENAMES = ("se_whoosh.wav", "se_impact.wav", "se_tick.wav")


def _write_wav(path: str, samples) -> None:
    """[-1, 1] のfloat列を16bitモノラルPCMのWAVとして書き出す。"""
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _lowpass_sweep(noise, start_coef: float, end_coef: float):
    """1極ローパスの係数を start→end へ動かしながら通す。係数が大きいほど暗い音。
    係数を動かすことで「シュッ」と抜ける感じ（whoosh）になる。"""
    import numpy as np

    coefs = np.linspace(start_coef, end_coef, len(noise))
    out = np.empty_like(noise)
    prev = 0.0
    for i in range(len(noise)):
        a = coefs[i]
        prev = a * prev + (1.0 - a) * noise[i]
        out[i] = prev
    return out


def _whoosh(duration: float = 0.25):
    import numpy as np

    n = int(SAMPLE_RATE * duration)
    # 乱数を固定して、同じ波形のファイルが毎回できるようにする（差分レビューのため）
    noise = np.random.default_rng(20260819).normal(0.0, 1.0, n)
    swept = _lowpass_sweep(noise, 0.995, 0.90)
    t = np.linspace(0.0, 1.0, n)
    envelope = np.sin(math.pi * t) ** 2  # 立ち上がりも切れ際もなめらか
    peak = float(np.max(np.abs(swept))) or 1.0
    return swept / peak * envelope * 0.9


def _impact(duration: float = 0.30):
    import numpy as np

    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE
    # 80Hzの胴鳴り。頭だけ倍音を足して輪郭を出す
    body = np.sin(2 * math.pi * 80 * t) + 0.35 * np.sin(2 * math.pi * 160 * t)
    return body * np.exp(-t * 14.0) * 0.9


def _tick(duration: float = 0.04):
    import numpy as np

    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE
    return np.sin(2 * math.pi * 1200 * t) * np.exp(-t * 90.0) * 0.8


def ensure_sound_effects(out_dir: str) -> bool:
    """SEのwavを out_dir に書き出す。numpy が無い・書き出しに失敗した場合は False
    （呼び出し側はSE無しでレンダリングを続ける。音のために投稿を止めない）。"""
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[se] numpy が無いため効果音をスキップします")
        return False

    try:
        os.makedirs(out_dir, exist_ok=True)
        _write_wav(os.path.join(out_dir, "se_whoosh.wav"), _whoosh())
        _write_wav(os.path.join(out_dir, "se_impact.wav"), _impact())
        _write_wav(os.path.join(out_dir, "se_tick.wav"), _tick())
        return True
    except Exception as e:
        print(f"[se] 効果音の生成に失敗しました: {e}")
        return False

"""
video/audio_gen.py

動画に乗せる効果音とBGMを numpy で自前生成する。

カット頭の無音が「演出ではなく再生バグに聞こえる」「BGMが無いと未完成品に見える」という
指摘が 2026-08-19 のインフルエンサーレビューで最多だった。ただしフリー素材を毎日
ダウンロードする方式だとライセンス確認が自動化できず、規約変更にも気づけないため、
波形をその場で合成する（著作権が発生しないので全自動運用と両立する）。

- se_whoosh.wav: カット頭。ローパスを開きながらのノイズスイープ（0.25秒）
- se_impact.wav: 金額の着地。低域のドン（0.30秒）
- se_tick.wav  : 数字のカウントアップ完了。短いチック（0.04秒）
- bgm.wav      : 全編に敷くアンビエントのパッド（12秒・継ぎ目なしでループ）

numpy が無い環境では False を返し、呼び出し側は音の追加なしで動画を書き出す。
"""
import math
import os
import wave

SAMPLE_RATE = 44100

# render.py が public/ へ書き出し、レンダリング後に消すファイル名。
GENERATED_AUDIO = ("se_whoosh.wav", "se_impact.wav", "se_tick.wav", "bgm.wav")

# BGMのループ長（秒）。4和音×3秒。40秒の動画で3周ちょっと。
BGM_LOOP_SEC = 12.0
BGM_CHORD_SEC = 3.0

# A マイナーの Am → F → C → G。金融の解説に乗せても主張しすぎない並び。
# 各和音は [ベース, 構成音...]（Hz、A4=440基準）。
BGM_CHORDS = (
    (110.00, 220.00, 261.63, 329.63),  # Am
    (87.31, 174.61, 220.00, 261.63),   # F
    (130.81, 261.63, 329.63, 392.00),  # C
    (98.00, 196.00, 246.94, 293.66),   # G
)


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


def _lowpass(signal, coef: float):
    """1極ローパス（係数固定）。合成音のざらつきを丸めて耳当たりを良くする。"""
    import numpy as np

    out = np.empty_like(signal)
    prev = 0.0
    for i in range(len(signal)):
        prev = coef * prev + (1.0 - coef) * signal[i]
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


def _pad_envelope(n: int):
    """パッド1音ぶんの音量カーブ。ゆっくり立ち上がってゆっくり消える。
    和音の枠（3秒）より長くして次の和音に被せることで、切れ目のない持続音になる。"""
    import numpy as np

    attack = int(SAMPLE_RATE * 0.8)
    release = int(SAMPLE_RATE * 1.4)
    env = np.ones(n)
    # 生の直線ではなくコサインカーブにする（直線だと立ち上がりが機械的に聞こえる）
    env[:attack] = (1 - np.cos(np.linspace(0, math.pi, attack))) / 2
    env[n - release:] = (1 + np.cos(np.linspace(0, math.pi, release))) / 2
    return env


def _bgm():
    """12秒で継ぎ目なくループするアンビエントのパッド。

    和音の音は枠（3秒）をはみ出す長さで鳴らし、はみ出したぶんは配列の先頭へ
    回り込ませて足す。こうするとループの継ぎ目に音の切れ目が来ない。"""
    import numpy as np

    n = int(SAMPLE_RATE * BGM_LOOP_SEC)
    buf = np.zeros(n)
    note_len = int(SAMPLE_RATE * (BGM_CHORD_SEC + 1.4))  # 次の和音に1.4秒被せる
    t = np.arange(note_len) / SAMPLE_RATE
    env = _pad_envelope(note_len)

    for i, chord in enumerate(BGM_CHORDS):
        start = int(SAMPLE_RATE * BGM_CHORD_SEC * i)
        idx = (start + np.arange(note_len)) % n  # 末尾を先頭へ回り込ませる
        for j, freq in enumerate(chord):
            # ベース音（先頭）は倍音を抑えて土台に、上の音は薄く重ねる
            gain = 0.9 if j == 0 else 0.45
            wave_ = np.sin(2 * math.pi * freq * t)
            wave_ += 0.35 * np.sin(2 * math.pi * freq * 2 * t)
            wave_ += 0.12 * np.sin(2 * math.pi * freq * 3 * t)
            # わずかにずらした音を重ねて厚みを出す（コーラス効果）
            wave_ += 0.5 * np.sin(2 * math.pi * freq * 1.003 * t)
            np.add.at(buf, idx, wave_ * env * gain)

    # ゆっくりした音量の揺れ。ループ長で割り切れる周期にして継ぎ目を作らない
    lfo_t = np.arange(n) / SAMPLE_RATE
    buf *= 1.0 - 0.12 * (1 - np.cos(2 * math.pi * (3 / BGM_LOOP_SEC) * lfo_t)) / 2

    # ローパスは内部状態が0から始まるため、そのまま掛けると先頭だけ音が痩せて
    # ループの継ぎ目に段差ができる。2周ぶん通してから後半だけを採り、
    # フィルタが定常状態になった区間を使う。
    buf = _lowpass(np.concatenate([buf, buf]), 0.55)[n:]
    peak = float(np.max(np.abs(buf))) or 1.0
    return buf / peak * 0.55


def ensure_sound_effects(out_dir: str) -> bool:
    """効果音とBGMのwavを out_dir に書き出す。numpy が無い・書き出しに失敗した場合は False
    （呼び出し側は音の追加なしでレンダリングを続ける。音のために投稿を止めない）。"""
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[audio_gen] numpy が無いため効果音とBGMをスキップします")
        return False

    try:
        os.makedirs(out_dir, exist_ok=True)
        _write_wav(os.path.join(out_dir, "se_whoosh.wav"), _whoosh())
        _write_wav(os.path.join(out_dir, "se_impact.wav"), _impact())
        _write_wav(os.path.join(out_dir, "se_tick.wav"), _tick())
        _write_wav(os.path.join(out_dir, "bgm.wav"), _bgm())
        return True
    except Exception as e:
        print(f"[audio_gen] 効果音・BGMの生成に失敗しました: {e}")
        return False

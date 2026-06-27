---
name: manim-video
description: Manim（Mathematical Animation Engine）を使ってPythonで美しいアニメーション動画を生成する。動画作成・アニメーション・可視化動画・Manim・数学アニメを求められたときに使う。
---

# Manim 動画生成

Manim（Community Edition）を使って Python でアニメーション動画を生成する。
参考: https://qiita.com/Kohki_Mametani/items/3c7a44e2445958a5cc25

## セットアップ

### 1. システム依存パッケージ（初回のみ）

```bash
sudo apt-get update && sudo apt-get install -y \
  libcairo2-dev libpango1.0-dev ffmpeg \
  texlive texlive-latex-extra texlive-fonts-extra texlive-latex-recommended \
  texlive-science tipa libgl1-mesa-glx
```

LaTeX が不要なら `texlive` 系は省略可（`MathTex` を使わない場合）。

### 2. Python パッケージ

```bash
pip install manim
```

## 基本的な使い方

### シーンの作成

Manim はシーン（`Scene` クラス）単位で動画を構成する。

```python
from manim import *

class MyScene(Scene):
    def construct(self):
        # オブジェクト作成
        circle = Circle(radius=2, color=BLUE)
        text = Text("Hello Manim!", font_size=48)

        # アニメーション
        self.play(Create(circle))
        self.play(Write(text))
        self.wait(1)
```

### レンダリング

```bash
# 低画質プレビュー（速い）
manim -ql scene.py MyScene

# 中画質
manim -qm scene.py MyScene

# 高画質（1080p）
manim -qh scene.py MyScene

# 4K
manim -qk scene.py MyScene

# GIF出力
manim -ql --format gif scene.py MyScene
```

出力先: `media/videos/<ファイル名>/<画質>/` に `.mp4` が生成される。

## よく使うオブジェクト

| クラス | 用途 | 例 |
|--------|------|-----|
| `Text` | テキスト表示 | `Text("Hello", font_size=36)` |
| `MathTex` | LaTeX数式 | `MathTex(r"\sum_{i=1}^n x_i")` |
| `Circle`, `Square`, `Rectangle` | 図形 | `Circle(radius=1, color=RED)` |
| `Arrow`, `Line` | 矢印・線 | `Arrow(LEFT, RIGHT)` |
| `Axes` | グラフ軸 | `Axes(x_range=[0,10], y_range=[0,5])` |
| `BarChart` | 棒グラフ | `BarChart(values=[1,3,2])` |
| `NumberPlane` | 座標平面 | `NumberPlane()` |
| `Table` | 表 | `Table([["A","B"],["1","2"]])` |
| `VGroup` | グループ化 | `VGroup(circle, text).arrange(DOWN)` |

## よく使うアニメーション

| メソッド | 効果 |
|----------|------|
| `Create(mob)` | 描画して出現 |
| `Write(mob)` | 書くように出現（テキスト向き） |
| `FadeIn(mob)` / `FadeOut(mob)` | フェードイン/アウト |
| `Transform(a, b)` | a を b に変形 |
| `ReplacementTransform(a, b)` | a を b に置換変形 |
| `Indicate(mob)` | 強調（光る） |
| `mob.animate.shift(RIGHT)` | 移動アニメーション |
| `mob.animate.scale(2)` | 拡大アニメーション |
| `mob.animate.set_color(RED)` | 色変更アニメーション |
| `AnimationGroup(*anims)` | 複数同時再生 |
| `Succession(*anims)` | 連続再生 |

## 実用パターン

### グラフアニメーション（株価チャート等）

```python
class StockChart(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 30, 5],
            y_range=[0, 100, 20],
            axis_config={"include_numbers": True},
        )
        labels = axes.get_axis_labels(x_label="日数", y_label="株価")

        # データ点をプロット
        points = [axes.c2p(x, y) for x, y in enumerate(prices)]
        graph = VMobject().set_points_smoothly(points)
        graph.set_color(GREEN)

        self.play(Create(axes), Write(labels))
        self.play(Create(graph), run_time=3)
        self.wait(1)
```

### 棒グラフの比較

```python
class CompareBar(Scene):
    def construct(self):
        chart = BarChart(
            values=[3.2, 5.1, 2.8, 4.5],
            bar_names=["Q1", "Q2", "Q3", "Q4"],
            y_range=[0, 6, 1],
            bar_colors=[BLUE, GREEN, RED, YELLOW],
        )
        self.play(Create(chart))
        self.wait(1)
```

### テキストの順次表示

```python
class BulletPoints(Scene):
    def construct(self):
        title = Text("分析結果", font_size=48, color=YELLOW)
        items = VGroup(
            Text("・勝率: 68%", font_size=32),
            Text("・平均リターン: +5.2%", font_size=32),
            Text("・最大DD: -8.1%", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        VGroup(title, items).arrange(DOWN, buff=1)

        self.play(Write(title))
        for item in items:
            self.play(FadeIn(item, shift=RIGHT), run_time=0.5)
        self.wait(2)
```

## 注意事項

- 日本語フォントは環境依存。`Text("日本語", font="Noto Sans CJK JP")` のようにフォント指定が必要な場合がある。
- `MathTex` は LaTeX が必要。インストールされていないと `MathTex` 使用時にエラーになる。
- レンダリングは CPU 負荷が高い。プレビューには `-ql`（低画質）を使う。
- `self.wait(秒)` で停止時間を制御。省略すると即座に次のアニメーションに移る。
- 色は定数（`RED`, `BLUE`, `GREEN`, `YELLOW`, `WHITE` 等）または hex（`"#FF5733"`）で指定。

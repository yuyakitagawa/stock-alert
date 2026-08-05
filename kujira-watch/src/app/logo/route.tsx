import { ImageResponse } from "next/og";

// Organization/ArticleのJSON-LDが参照する正方形ロゴ画像（構造化データにはOGP画像のような
// 横長比率ではなく正方形〜近い比率の画像を指定するのがGoogleの推奨のため専用に用意する）。
export const size = { width: 512, height: 512 };
export const contentType = "image/png";

export async function GET() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#16213a",
          fontSize: 280,
        }}
      >
        🐋
      </div>
    ),
    { ...size }
  );
}

import { ImageResponse } from "next/og";

// iOS「ホーム画面に追加」用。iOS側で角丸が付くので背景は正方形で塗る
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
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
          fontSize: 120,
        }}
      >
        🐋
      </div>
    ),
    { ...size }
  );
}

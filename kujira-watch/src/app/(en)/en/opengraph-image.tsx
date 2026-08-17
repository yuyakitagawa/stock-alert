import { ImageResponse } from "next/og";
import { SITE_DESCRIPTION_EN, SITE_NAME_EN } from "@/lib/site";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "#16213a",
          padding: 80,
          textAlign: "center",
        }}
      >
        <div style={{ display: "flex", fontSize: 140, marginBottom: 24 }}>🐋</div>
        <div style={{ display: "flex", fontSize: 68, fontWeight: 700, color: "#ffffff" }}>
          {SITE_NAME_EN}
        </div>
        <div
          style={{
            display: "flex",
            marginTop: 20,
            fontSize: 16,
            fontWeight: 700,
            letterSpacing: 4,
            textTransform: "uppercase",
            color: "#d9a44f",
          }}
        >
          Kujira Watch
        </div>
        <div
          style={{
            display: "flex",
            marginTop: 28,
            fontSize: 28,
            color: "rgba(255,255,255,0.75)",
            maxWidth: 900,
          }}
        >
          {SITE_DESCRIPTION_EN}
        </div>
      </div>
    ),
    { ...size }
  );
}

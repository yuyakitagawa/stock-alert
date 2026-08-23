import { ImageResponse } from "next/og";

export const contentType = "image/png";

const SIZES = [32, 192, 512];

export function generateImageMetadata() {
  return SIZES.map((s) => ({
    id: String(s),
    size: { width: s, height: s },
    contentType,
  }));
}

export default async function Icon({ id }: { id: Promise<string> }) {
  const s = Number(await id);
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
          borderRadius: "50%",
          fontSize: Math.round(s * 0.68),
        }}
      >
        🐋
      </div>
    ),
    { width: s, height: s }
  );
}

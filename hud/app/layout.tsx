import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Inside the Agent",
  description: "Live interpretability HUD for SAE-steered language agents",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

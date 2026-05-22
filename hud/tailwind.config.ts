import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        risk: {
          high: "#dc2626",
          med: "#ca8a04",
          low: "#16a34a",
        },
      },
    },
  },
  plugins: [],
};

export default config;

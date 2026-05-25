import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#08131f",
        mist: "#e8f1f8",
        glow: "#77d6ff",
        ember: "#ff9a62",
        slate: "#9ab4c7"
      },
      boxShadow: {
        panel: "0 24px 70px rgba(3, 12, 20, 0.28)"
      },
      animation: {
        "fade-up": "fadeUp 480ms ease-out both",
        pulsegrid: "pulseGrid 8s ease-in-out infinite"
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(14px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        pulseGrid: {
          "0%, 100%": { opacity: "0.35" },
          "50%": { opacity: "0.75" }
        }
      }
    }
  },
  plugins: []
};

export default config;

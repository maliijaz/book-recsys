import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    // goodbooks-10k cover art is served from Goodreads' asset CDN.
    remotePatterns: [
      { protocol: "https", hostname: "images.gr-assets.com" },
      { protocol: "https", hostname: "s.gr-assets.com" },
    ],
  },
};

export default nextConfig;

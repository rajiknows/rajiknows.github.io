// @ts-check
import fs from 'node:fs';
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import vercel from '@astrojs/vercel';

// Gruvbox isn't in Shiki's bundled theme set, so load our local copies.
const gruvboxLight = JSON.parse(
  fs.readFileSync(new URL('./src/styles/themes/gruvbox-light.json', import.meta.url), 'utf-8'),
);
const gruvboxDark = JSON.parse(
  fs.readFileSync(new URL('./src/styles/themes/gruvbox-dark.json', import.meta.url), 'utf-8'),
);

// https://astro.build/config
export default defineConfig({
  site: 'https://rajiknows.github.io',
  output: 'static',
  adapter: vercel({
    webAnalytics: {
      enabled: true,
    },
  }),
  markdown: {
    shikiConfig: {
      // Multi-theme: each key emits its own --shiki-<key> CSS vars per token.
      // site.css picks the active set by the [data-theme] attribute.
      // Keys must match the data-theme values in site.css / ThemeToggle.astro.
      // Gruvbox isn't bundled in Shiki, so we pass the local JSON objects.
      themes: {
        "gruvbox-light": gruvboxLight,
        "gruvbox-dark": gruvboxDark,
        catppuccin: "catppuccin-mocha",
        tokyonight: "tokyo-night",
        kanagawa: "kanagawa-wave",
      },
      // Emit CSS variables only (no inline default colors) so highlighting
      // follows the theme picker rather than prefers-color-scheme.
      defaultColor: false,
    },
  },
  integrations: [mdx(), sitemap()],
});
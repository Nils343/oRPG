import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['static/tests/**/*.test.js'],
    globals: false,
  },
});

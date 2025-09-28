import { describe, expect, it } from 'vitest';

import { __installStubsForTests } from './test-utils.js';

describe('test-utils installStubs', () => {
  it('polyfills expected browser APIs when missing', async () => {
    const fakeWindow = {
      URL,
      location: { href: 'https://example.com/' },
      HTMLMediaElement: function HTMLMediaElement() {},
      setTimeout,
      clearTimeout,
    };
    fakeWindow.HTMLMediaElement.prototype = {};

    __installStubsForTests(fakeWindow);

    expect(typeof fakeWindow.matchMedia).toBe('function');
    expect(typeof fakeWindow.requestAnimationFrame).toBe('function');
    expect(typeof fakeWindow.cancelAnimationFrame).toBe('function');

    const audio = new fakeWindow.Audio();
    expect(audio.paused).toBe(true);
    await audio.play();
    expect(audio.paused).toBe(false);

    const response = await fakeWindow.fetch('/api/state');
    expect(typeof response.json).toBe('function');

    fakeWindow.requestAnimationFrame(() => {});
    fakeWindow.cancelAnimationFrame(0);
  });
});

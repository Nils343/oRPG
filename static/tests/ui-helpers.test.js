import { describe, it, expect, vi } from 'vitest';
import { createClientDom } from './test-utils.js';

describe('Join backgrounds', () => {
  it('sanitizes remote backgrounds and applies the first valid option', async () => {
    const { window, document, cleanup } = await createClientDom();
    const originalFetch = window.fetch;
    const randomSpy = vi.spyOn(window.Math, 'random').mockReturnValue(0);
    window.fetch = vi.fn(async (input, init) => {
      if (typeof input === 'string' && input.endsWith('/api/join_backgrounds')) {
        return {
          ok: true,
          json: async () => ({ backgrounds: ['   ', ' /static/img/test-bg.png ', null, '/static/img/other.png'] }),
        };
      }
      return originalFetch(input, init);
    });

    try {
      await window.loadJoinBackgrounds();
      const bg = document.body.style.getPropertyValue('--join-screen-bg');
      expect(bg).toBe('url("/static/img/test-bg.png")');
    } finally {
      window.fetch = originalFetch;
      randomSpy.mockRestore();
      cleanup();
    }
  });
});

describe('FX saver toggle', () => {
  it('updates the root class and asks for persistence when requested', async () => {
    const { window, document, cleanup } = await createClientDom();
    const persistSpy = vi.spyOn(window, 'persistFxPreference');

    try {
      window.applyFxSaver(false, { persist: true });
      expect(document.documentElement.classList.contains('nofx')).toBe(false);
      expect(persistSpy).toHaveBeenCalledWith(false);

      window.applyFxSaver(true, { persist: true });
      expect(document.documentElement.classList.contains('nofx')).toBe(true);
      expect(persistSpy).toHaveBeenCalledWith(true);
    } finally {
      persistSpy.mockRestore();
      cleanup();
    }
  });
});

describe('FX preference persistence', () => {
  it('writes the stored value when storage is available', async () => {
    const { window, cleanup } = await createClientDom();
    const storage = window.localStorage;
    const storageProto = storage ? Object.getPrototypeOf(storage) : null;
    if (!storageProto) {
      throw new Error('localStorage prototype unavailable');
    }
    const setItemSpy = vi.spyOn(storageProto, 'setItem');

    try {
      window.persistFxPreference(true);
      window.persistFxPreference(false);
      expect(setItemSpy).toHaveBeenNthCalledWith(1, 'orpg-nofx', '1');
      expect(setItemSpy).toHaveBeenNthCalledWith(2, 'orpg-nofx', '0');
    } finally {
      setItemSpy.mockRestore();
      cleanup();
    }
  });
});

describe('World style loading', () => {
  it('falls back to the default tree when the payload is invalid', async () => {
    const { window, cleanup } = await createClientDom();
    await window.ensureWorldStylesInitialized();
    expect(window.worldStyleExists('Solarpunk')).toBe(true);

    const originalFetch = window.fetch;
    const warnSpy = vi.spyOn(window.console, 'warn').mockImplementation(() => {});
    window.fetch = vi.fn(async (input, init) => {
      if (typeof input === 'string' && input.includes('world-styles.json')) {
        return { ok: true, json: async () => ({}) };
      }
      return originalFetch(input, init);
    });

    try {
      await window.loadWorldStyleData();
      expect(window.worldStyleExists('Solarpunk')).toBe(false);
      expect(window.worldStyleExists('High Fantasy')).toBe(true);
      expect(warnSpy).toHaveBeenCalled();
    } finally {
      window.fetch = originalFetch;
      warnSpy.mockRestore();
      cleanup();
    }
  });
});

describe('Join music library', () => {
  it('filters invalid songs and stops playback when nothing remains', async () => {
    const { window, cleanup } = await createClientDom();
    const originalStop = window.stopJoinMusic;
    const stopSpy = vi.fn();
    window.stopJoinMusic = stopSpy;

    try {
      window.setJoinMusicLibrary([
        null,
        {},
        { id: ' Theme ', src: ' /static/audio/theme.mp3 ' },
        { id: 'secondary', src: ' ' },
      ]);
      expect(window.joinMusicAvailable()).toBe(true);
      window.resetJoinMusicQueue();
      const track = window.nextJoinTrack();
      expect(track).toEqual({ id: 'Theme', src: '/static/audio/theme.mp3', title: 'Theme' });

      stopSpy.mockClear();
      window.setJoinMusicLibrary([{ id: ' ', src: ' ' }]);
      expect(window.joinMusicAvailable()).toBe(false);
      expect(stopSpy).toHaveBeenCalledWith({ resetQueue: true });
    } finally {
      window.stopJoinMusic = originalStop;
      cleanup();
    }
  });
});

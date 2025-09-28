import { describe, it, expect, vi } from 'vitest';
import { createClientDom } from './test-utils.js';

describe('Narrative history scrolling', () => {
  it('only auto-scrolls when pinned near the bottom and maintains history blocks', async () => {
    const { window, document, flush, cleanup } = await createClientDom();
    const narrative = document.getElementById('narrative');
    if (!narrative) {
      throw new Error('narrative element missing');
    }

    const sizer = window.narrativeSizer;
    const originalResize = sizer ? sizer.resize : null;
    const originalRaf = window.requestAnimationFrame;
    const originalCancelRaf = window.cancelAnimationFrame;
    window.requestAnimationFrame = (cb) => {
      cb(Date.now());
      return 0;
    };
    window.cancelAnimationFrame = () => {};
    if (sizer && sizer._mutationObserver) {
      sizer._mutationObserver.disconnect();
      sizer._mutationObserver = null;
    }
    if (sizer && sizer._resizeObserver && typeof sizer._resizeObserver.disconnect === 'function') {
      sizer._resizeObserver.disconnect();
      sizer._resizeObserver = null;
    }
    if (sizer) {
      sizer.resize = () => {};
    }

    let scrollTopValue = 0;
    Object.defineProperty(narrative, 'scrollTop', {
      configurable: true,
      get: () => scrollTopValue,
      set: (value) => { scrollTopValue = value; },
    });
    Object.defineProperty(narrative, 'clientHeight', {
      configurable: true,
      get: () => 300,
    });
    Object.defineProperty(narrative, 'scrollHeight', {
      configurable: true,
      get: () => 900,
    });

    try {
      window.renderNarrativeHistory([
        { turn: 2, narrative: 'Second turn text', timestamp: 200 },
        { turn: 1, narrative: '', timestamp: 100 },
      ]);

      const initialBlocks = Array.from(narrative.querySelectorAll('.narrative-block'));
      expect(initialBlocks.length).toBe(2);
      const initialByTurn = Object.fromEntries(initialBlocks.map((block) => [block.dataset.turn, block.textContent]));
      expect(initialByTurn['2']).toBe('Second turn text');
      expect(initialByTurn['1']).toBe('—');

      scrollTopValue = 80; // beyond the pin threshold
      const scrollSpy = vi.spyOn(window, 'scrollNarrativeToTop');

      window.renderPublic({
        turn_index: 3,
        current_narrative: 'Third turn narrative',
        history: [
          { turn: 3, narrative: 'Third turn narrative', timestamp: 300 },
          { turn: 2, narrative: 'Second turn text', timestamp: 200 },
        ],
        history_summary: [],
        players: [],
        auto_image_enabled: false,
        auto_video_enabled: false,
        auto_tts_enabled: false,
        lock: { active: false },
        token_usage: {},
      });
      await flush();

      expect(scrollSpy).not.toHaveBeenCalled();
      expect(scrollTopValue).toBe(80);

      scrollSpy.mockClear();
      scrollTopValue = 12; // within the pin threshold, should auto-scroll

      window.renderPublic({
        turn_index: 4,
        current_narrative: 'Fourth turn narrative',
        history: [
          { turn: 4, narrative: 'Fourth turn narrative', timestamp: 400 },
          { turn: 3, narrative: 'Third turn narrative', timestamp: 300 },
        ],
        history_summary: [],
        players: [],
        auto_image_enabled: false,
        auto_video_enabled: false,
        auto_tts_enabled: false,
        lock: { active: false },
        token_usage: {},
      });
      await flush();
      await flush();

      expect(scrollSpy).toHaveBeenCalled();
      expect(scrollTopValue).toBe(0);

      const updatedBlocks = Array.from(narrative.querySelectorAll('.narrative-block'));
      expect(updatedBlocks.length).toBe(2);
      const updatedByTurn = Object.fromEntries(updatedBlocks.map((block) => [block.dataset.turn, block.textContent]));
      expect(updatedByTurn['4']).toBe('Fourth turn narrative');
      expect(updatedByTurn['1']).toBeUndefined();
    } finally {
      if (sizer && originalResize) {
        sizer.resize = originalResize;
      }
      window.requestAnimationFrame = originalRaf;
      window.cancelAnimationFrame = originalCancelRaf;
      vi.restoreAllMocks();
      cleanup();
    }
  });
});

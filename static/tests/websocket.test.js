import { describe, it, expect, vi } from 'vitest';
import { createClientDom } from './test-utils.js';

function disableNarrativeSizer(win) {
  const sizer = win.narrativeSizer;
  if (!sizer) return;
  if (sizer._mutationObserver) {
    sizer._mutationObserver.disconnect();
    sizer._mutationObserver = null;
  }
  if (sizer._resizeObserver && typeof sizer._resizeObserver.disconnect === 'function') {
    sizer._resizeObserver.disconnect();
    sizer._resizeObserver = null;
  }
}

describe('WebSocket wiring', () => {
  it('routes state and private events through the renderers', async () => {
    const { window, flush, cleanup } = await createClientDom();
    const OriginalWebSocket = window.WebSocket;
    let capturedSocket = null;

    class TrackingWebSocket extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        capturedSocket = this;
      }
    }

    window.WebSocket = TrackingWebSocket;
    disableNarrativeSizer(window);
    const { document } = window;

    try {
      const publicSpy = vi.spyOn(window, 'renderPublic');
      const privateSpy = vi.spyOn(window, 'renderPrivate');

      window.connectWS();

      expect(capturedSocket, 'connectWS should create a WebSocket instance').toBeTruthy();

      const statePayload = {
        turn_index: 5,
        current_narrative: 'Turn five narrative',
        history: [
          { turn: 5, narrative: 'Turn five narrative', timestamp: 555 },
        ],
        history_summary: [],
        players: [],
        auto_image_enabled: true,
        auto_video_enabled: false,
        auto_tts_enabled: true,
        language: 'en',
        token_usage: {},
        lock: { active: false },
      };

      capturedSocket.onmessage({ data: JSON.stringify({ event: 'state', data: statePayload }) });
      await flush();

      expect(publicSpy).toHaveBeenCalledWith(statePayload);
      const turnHeader = document.getElementById('turnHeader');
      expect(turnHeader?.textContent).toBe(window.t('game.turn', { count: 5 }));
      const narrativeBlocks = Array.from(document.querySelectorAll('#narrative .narrative-block'));
      expect(narrativeBlocks.some((block) => block.textContent === 'Turn five narrative')).toBe(true);

      const privatePayload = {
        you: {
          id: 'player-5',
          name: 'Heroine',
          pending_join: true,
          abilities: [],
          inventory: [],
          conditions: [],
        },
      };

      capturedSocket.onmessage({ data: JSON.stringify({ event: 'private', data: privatePayload }) });
      await flush();

      expect(privateSpy).toHaveBeenCalledWith(privatePayload);
      const youFlags = document.getElementById('youFlags');
      expect(youFlags?.textContent).toContain(window.t('game.playerFlag.queued'));
    } finally {
      window.WebSocket = OriginalWebSocket;
      vi.restoreAllMocks();
      cleanup();
    }
  });
});

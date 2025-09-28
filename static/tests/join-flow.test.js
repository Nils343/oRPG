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

describe('Join flow behaviour', () => {
  it('toggles form interactivity when waiting to join', async () => {
    const { window, document, cleanup } = await createClientDom();
    disableNarrativeSizer(window);
    try {
      const enterButton = document.getElementById('btnEnter');
      const nameInput = document.getElementById('name');
      const backgroundInput = document.getElementById('background');

      expect(enterButton, 'join button should exist').toBeTruthy();
      expect(nameInput, 'name input should exist').toBeTruthy();
      expect(backgroundInput, 'background input should exist').toBeTruthy();

      // Initial state comes from initClient -> setJoinWaiting(false)
      expect(enterButton.hasAttribute('disabled')).toBe(false);
      expect(nameInput.hasAttribute('disabled')).toBe(false);
      expect(backgroundInput.hasAttribute('disabled')).toBe(false);

      window.setJoinWaiting(true);

      expect(enterButton.getAttribute('disabled')).toBe('true');
      expect(nameInput.getAttribute('disabled')).toBe('true');
      expect(backgroundInput.getAttribute('disabled')).toBe('true');

      window.setJoinWaiting(false);

      expect(enterButton.hasAttribute('disabled')).toBe(false);
      expect(nameInput.hasAttribute('disabled')).toBe(false);
      expect(backgroundInput.hasAttribute('disabled')).toBe(false);
    } finally {
      cleanup();
    }
  });

  it('updates join status visibility based on hidden keys', async () => {
    const { window, document, cleanup } = await createClientDom();
    disableNarrativeSizer(window);
    const statusEl = document.getElementById('joinStatus');
    if (!statusEl) {
      throw new Error('joinStatus element missing');
    }
    statusEl.style.display = '';
    statusEl.textContent = 'Existing';

    const tSpy = vi.spyOn(window, 't').mockImplementation((key) => `translated:${key}`);

    try {
      window.updateJoinStatus('', { key: 'join.status.queueActive' });

      expect(tSpy).not.toHaveBeenCalled();
      expect(statusEl.textContent).toBe('');
      expect(statusEl.style.display).toBe('none');

      tSpy.mockClear();
      const args = { seconds: 5 };
      window.updateJoinStatus('', { key: 'join.status.invalidSession', args });

      expect(tSpy).toHaveBeenCalledWith('join.status.invalidSession', args);
      expect(statusEl.textContent).toBe('translated:join.status.invalidSession');
      expect(statusEl.style.display).toBe('');
    } finally {
      vi.restoreAllMocks();
      cleanup();
    }
  });

  it('persists pending join state and clears countdown when resolved', async () => {
    const { window, document, flush, cleanup } = await createClientDom();
    disableNarrativeSizer(window);
    try {
      window.localStorage.setItem('orpg-player-session', JSON.stringify({
        id: 'player-123',
        token: 'token-abc',
        name: 'Hero',
        background: 'Backstory',
      }));

      const connectStub = vi.spyOn(window, 'connectWS').mockImplementation(() => {});
      window.initClient();
      await flush();
      connectStub.mockRestore();

      const enterButton = document.getElementById('btnEnter');
      const countdownSpy = vi.spyOn(window, 'stopButtonCountdown');

      window.handleJoinState(true);

      expect(enterButton?.getAttribute('disabled')).toBe('true');
      expect(countdownSpy).not.toHaveBeenCalled();

      const storedPending = window.getStoredPlayer();
      expect(storedPending).toMatchObject({
        id: 'player-123',
        token: 'token-abc',
        name: 'Hero',
        background: 'Backstory',
        pending: true,
      });

      countdownSpy.mockClear();
      window.handleJoinState(false);

      expect(enterButton?.hasAttribute('disabled')).toBe(false);
      expect(countdownSpy).toHaveBeenCalledWith('btnEnter');

      const storedResolved = window.getStoredPlayer();
      expect(storedResolved).toMatchObject({
        id: 'player-123',
        token: 'token-abc',
        name: 'Hero',
        background: 'Backstory',
        pending: false,
      });

    } finally {
      vi.restoreAllMocks();
      cleanup();
    }
  });

  it('expires the session, resets state, and alerts the user', async () => {
    const { window, document, flush, cleanup } = await createClientDom();
    disableNarrativeSizer(window);
    const nameInput = document.getElementById('name');
    const backgroundInput = document.getElementById('background');
    if (!nameInput || !backgroundInput) {
      throw new Error('join inputs missing');
    }

    try {
      window.localStorage.setItem('orpg-player-session', JSON.stringify({
        id: 'player-123',
        token: 'token-abc',
        name: 'Stored Hero',
        background: 'Stored Story',
        pending: true,
      }));

      const connectStub = vi.spyOn(window, 'connectWS').mockImplementation(() => {});
      window.initClient();
      await flush();
      connectStub.mockRestore();

      nameInput.value = 'Typed Hero';
      backgroundInput.value = 'Typed Story';
      window.handleJoinState(true);

      const expectedAlert = window.t('join.status.invalidSession');
      const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
      const updateSpy = vi.spyOn(window, 'updateJoinStatus');
      const joinWaitingSpy = vi.spyOn(window, 'setJoinWaiting');
      const closeSpy = vi.spyOn(window.WebSocket.prototype, 'close');

      // Create a live websocket instance so expireSession sees an open connection
      window.connectWS();

      window.expireSession('', { key: 'join.status.invalidSession' });

      expect(joinWaitingSpy).toHaveBeenCalledWith(false);
      expect(updateSpy).toHaveBeenCalledWith('', { key: 'join.status.invalidSession', args: undefined });
      expect(alertSpy).toHaveBeenCalledWith(expectedAlert);
      expect(closeSpy).toHaveBeenCalled();

      const stored = window.getStoredPlayer();
      expect(stored).toEqual({ name: 'Stored Hero', background: 'Stored Story' });
      expect(nameInput.value).toBe('Stored Hero');
      expect(backgroundInput.value).toBe('Stored Story');

    } finally {
      vi.restoreAllMocks();
      cleanup();
    }
  });
});

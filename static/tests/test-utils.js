import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HTML_PATH = path.resolve(__dirname, '..', 'index.html');
const HTML_SOURCE = readFileSync(HTML_PATH, 'utf8');

const WORLD_STYLE_PAYLOAD = {
  world_styles: [
    {
      name: 'Fantasy',
      children: [
        { name: 'High Fantasy' },
        { name: 'Solarpunk' },
      ],
    },
    {
      name: 'Science Fiction',
      children: [
        { name: 'Space Opera' },
      ],
    },
  ],
};

const DEFAULT_STATE_PAYLOAD = {
  turn_index: 0,
  current_narrative: '',
  history: [],
  history_summary: [],
  players: {},
  auto_image_enabled: false,
  auto_video_enabled: false,
  auto_tts_enabled: false,
};

const makeResponse = (data, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => data,
  text: async () => JSON.stringify(data),
});

const createFetchStub = (window) => (input, init = {}) => {
  const href = typeof input === 'string' ? input : input && typeof input.url === 'string' ? input.url : '';
  let pathname = href;
  try {
    const absolute = new window.URL(href, window.location.href);
    pathname = absolute.pathname;
  } catch (err) {
    /* ignore malformed URLs */
  }

  if (pathname === '/static/world-styles.json') {
    return Promise.resolve(makeResponse(WORLD_STYLE_PAYLOAD));
  }
  if (pathname === '/api/join_backgrounds') {
    return Promise.resolve(makeResponse({ backgrounds: [] }));
  }
  if (pathname === '/api/join_songs') {
    return Promise.resolve(makeResponse({ songs: [] }));
  }
  if (pathname === '/api/state') {
    return Promise.resolve(makeResponse(DEFAULT_STATE_PAYLOAD));
  }
  if (pathname === '/api/settings') {
    return Promise.resolve(makeResponse({
      world_style: 'High Fantasy',
      difficulty: 'Normal',
      thinking_mode: 'none',
    }));
  }
  if (pathname === '/api/join' && init && init.method === 'POST') {
    return Promise.resolve(makeResponse({
      player_id: 'test-player',
      auth_token: 'test-token',
    }));
  }
  if (pathname === '/api/language' && init && init.method === 'POST') {
    return Promise.resolve(makeResponse({ ok: true }));
  }
  if (pathname === '/api/submit' && init && init.method === 'POST') {
    return Promise.resolve(makeResponse({ ok: true }));
  }
  if (pathname === '/api/create_portrait' && init && init.method === 'POST') {
    return Promise.resolve(makeResponse({ ok: true }));
  }
  return Promise.resolve(makeResponse({}));
};

class FakeAudio {
  constructor(src = '') {
    this.src = src;
    this.loop = false;
    this.volume = 1;
    this.preload = 'auto';
    this.autoplay = false;
    this.paused = true;
  }

  play() {
    this.paused = false;
    return Promise.resolve();
  }

  pause() {
    this.paused = true;
  }

  addEventListener() {}

  removeEventListener() {}
}

class FakeWebSocket {
  constructor() {
    this.readyState = FakeWebSocket.OPEN;
    this.sent = [];
  }

  send(data) {
    this.sent.push(data);
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
  }
}

FakeWebSocket.CONNECTING = 0;
FakeWebSocket.OPEN = 1;
FakeWebSocket.CLOSING = 2;
FakeWebSocket.CLOSED = 3;

const installStubs = (window) => {
  window.fetch = createFetchStub(window);
  window.Audio = FakeAudio;
  if (window.HTMLMediaElement && window.HTMLMediaElement.prototype) {
    const proto = window.HTMLMediaElement.prototype;
    const originalPlay = proto.play;
    proto.play = function play(...args) {
      try {
        const result = originalPlay ? originalPlay.apply(this, args) : null;
        if (result && typeof result.then === 'function') {
          return result.catch(() => Promise.resolve());
        }
      } catch (err) {
        /* swallow media errors in tests */
      }
      this.paused = false;
      return Promise.resolve();
    };
    if (typeof proto.pause !== 'function') {
      proto.pause = function pause() {
        this.paused = true;
      };
    }
  }
  window.WebSocket = FakeWebSocket;
  if (typeof window.alert !== 'function') {
    window.alert = () => {};
  }
  if (typeof window.matchMedia !== 'function') {
    window.matchMedia = () => ({
      matches: false,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent() { return false; },
    });
  }
  if (typeof window.requestAnimationFrame !== 'function') {
    window.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 16);
  }
  if (typeof window.cancelAnimationFrame !== 'function') {
    window.cancelAnimationFrame = (id) => clearTimeout(id);
  }
};

const flushPromises = () => new Promise((resolve) => {
  setTimeout(resolve, 0);
});

export async function createClientDom() {
  const dom = new JSDOM(HTML_SOURCE, {
    url: 'https://example.com/',
    pretendToBeVisual: true,
    runScripts: 'dangerously',
    resources: 'usable',
    beforeParse(window) {
      installStubs(window);
    },
  });

  const { window } = dom;
  const { document } = window;

  if (document.readyState === 'loading') {
    await new Promise((resolve) => {
      document.addEventListener('DOMContentLoaded', resolve, { once: true });
    });
  }

  if (typeof window.ensureWorldStylesInitialized === 'function') {
    await window.ensureWorldStylesInitialized();
  }

  await flushPromises();

  return {
    dom,
    window,
    document,
    async flush() {
      await flushPromises();
    },
    cleanup() {
      window.close();
    },
  };
}

// Expose the stub installer for targeted unit tests.
export function __installStubsForTests(window) {
  installStubs(window);
}

export { flushPromises };

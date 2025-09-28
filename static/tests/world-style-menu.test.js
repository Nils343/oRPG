import { describe, it, expect } from 'vitest';
import { createClientDom, flushPromises } from './test-utils.js';

describe('World style picker', () => {
  it('reveals the custom input when the custom option is selected', async () => {
    const { window, document, cleanup } = await createClientDom();
    try {
      if (typeof window.ensureWorldStylesInitialized === 'function') {
        await window.ensureWorldStylesInitialized();
      }
      await flushPromises();

      const menu = document.getElementById('worldStyleMenu');
      const toggle = document.getElementById('worldStyleToggle');
      const select = document.getElementById('setWorld');
      const customInput = document.getElementById('setWorldCustom');

      expect(menu, 'world style menu should exist').toBeTruthy();
      expect(toggle, 'world style toggle should exist').toBeTruthy();
      expect(select, 'world style select input should exist').toBeTruthy();
      expect(customInput, 'custom world input should exist').toBeTruthy();
      expect(customInput.classList.contains('hidden')).toBe(true);

      window.selectWorldStyle('__custom__', { focusCustom: true, silent: true });

      expect(customInput.classList.contains('hidden')).toBe(false);
      expect(customInput.hasAttribute('disabled')).toBe(false);
      expect(document.activeElement).toBe(customInput);
      expect(select.value).toBe('__custom__');
      expect((toggle.textContent || '').toLowerCase()).toContain('custom');

      window.selectWorldStyle('High Fantasy', { silent: true });

      expect(customInput.classList.contains('hidden')).toBe(true);
      expect(toggle.getAttribute('data-current-value')).toBe('High Fantasy');
    } finally {
      cleanup();
    }
  });

  it('focuses the custom input when toggled on and disables it when hidden', async () => {
    const { window, document, cleanup } = await createClientDom();
    const customInput = document.getElementById('setWorldCustom');
    if (!customInput) {
      throw new Error('custom world input missing');
    }

    try {
      expect(customInput.classList.contains('hidden')).toBe(true);
      expect(customInput.hasAttribute('disabled')).toBe(true);

      window.toggleCustomWorldInput(true, { focus: true });

      expect(customInput.classList.contains('hidden')).toBe(false);
      expect(customInput.hasAttribute('disabled')).toBe(false);
      expect(document.activeElement).toBe(customInput);

      window.toggleCustomWorldInput(false);

      expect(customInput.classList.contains('hidden')).toBe(true);
      expect(customInput.getAttribute('disabled')).toBe('true');
    } finally {
      cleanup();
    }
  });

  it('updates the toggle label and hint when switching between grouped styles', async () => {
    const { window, document, cleanup } = await createClientDom();
    const menu = document.getElementById('worldStyleMenu');
    const toggle = document.getElementById('worldStyleToggle');
    const select = document.getElementById('setWorld');
    if (!menu || !toggle || !select) {
      throw new Error('world style controls missing');
    }

    try {
      window.applyWorldStyleTree([
        {
          name: 'Group Alpha',
          applies_to: 'any',
          children: [
            { name: 'Alpha Prime', applies_to: 'any' },
          ],
        },
      ]);
      window.renderWorldStyleMenu(menu);

      window.selectWorldStyle('Alpha Prime', { silent: true });

      expect(toggle.textContent).toBe('Alpha Prime');
      expect(toggle.getAttribute('data-current-value')).toBe('Alpha Prime');
      expect(toggle.title).toBe('Alpha Prime - Layer on any world');

      window.selectWorldStyle('Group Alpha', { silent: true });

      expect(toggle.textContent).toBe('Group Alpha');
      expect(toggle.getAttribute('data-current-value')).toBe('Group Alpha');
      expect(toggle.title).toBe('Group Alpha - Layer on any world');
      expect(select.value).toBe('Group Alpha');
    } finally {
      cleanup();
    }
  });
});

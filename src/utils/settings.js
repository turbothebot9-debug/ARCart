/**
 * Settings Manager - Handles app preferences and theme
 */

const DEFAULT_SETTINGS = {
  theme: 'dark',
  soundEnabled: true,
  voiceEnabled: false,
  hapticEnabled: true,
  scanAnimation: true
};

class SettingsManager {
  constructor() {
    this.settings = this.load();
    this.listeners = [];
  }

  load() {
    try {
      const saved = localStorage.getItem('arcart_settings');
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : { ...DEFAULT_SETTINGS };
    } catch (e) {
      return { ...DEFAULT_SETTINGS };
    }
  }

  save() {
    try {
      localStorage.setItem('arcart_settings', JSON.stringify(this.settings));
    } catch (e) {
      console.warn('Could not save settings:', e);
    }
  }

  get(key) {
    return this.settings[key];
  }

  set(key, value) {
    this.settings[key] = value;
    this.save();
    this.notify(key, value);
  }

  onChange(callback) {
    this.listeners.push(callback);
  }

  notify(key, value) {
    this.listeners.forEach(cb => cb(key, value));
  }

  // Apply theme to document
  applyTheme() {
    document.documentElement.setAttribute('data-theme', this.settings.theme);
  }

  // Toggle scan animation
  applyScanAnimation() {
    const scanLine = document.getElementById('scan-line');
    const scanCorners = document.querySelector('.scan-corners');
    
    if (scanLine) {
      scanLine.style.display = this.settings.scanAnimation ? 'block' : 'none';
    }
    if (scanCorners) {
      scanCorners.style.display = this.settings.scanAnimation ? 'block' : 'none';
    }
  }

  // Initialize settings UI
  initUI() {
    const themeSelect = document.getElementById('theme-select');
    const soundToggle = document.getElementById('sound-toggle');
    const voiceToggle = document.getElementById('voice-toggle');
    const hapticToggle = document.getElementById('haptic-toggle');
    const scanAnimationToggle = document.getElementById('scan-animation-toggle');

    // Set initial values
    if (themeSelect) themeSelect.value = this.settings.theme;
    if (soundToggle) soundToggle.checked = this.settings.soundEnabled;
    if (voiceToggle) voiceToggle.checked = this.settings.voiceEnabled;
    if (hapticToggle) hapticToggle.checked = this.settings.hapticEnabled;
    if (scanAnimationToggle) scanAnimationToggle.checked = this.settings.scanAnimation;

    // Add event listeners
    themeSelect?.addEventListener('change', (e) => {
      this.set('theme', e.target.value);
      this.applyTheme();
    });

    soundToggle?.addEventListener('change', (e) => {
      this.set('soundEnabled', e.target.checked);
    });

    voiceToggle?.addEventListener('change', (e) => {
      this.set('voiceEnabled', e.target.checked);
    });

    hapticToggle?.addEventListener('change', (e) => {
      this.set('hapticEnabled', e.target.checked);
    });

    scanAnimationToggle?.addEventListener('change', (e) => {
      this.set('scanAnimation', e.target.checked);
      this.applyScanAnimation();
    });

    // Apply initial settings
    this.applyTheme();
    this.applyScanAnimation();
  }
}

export const settings = new SettingsManager();
export default settings;

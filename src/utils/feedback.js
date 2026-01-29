/**
 * Feedback Manager - Handles sound, voice, and haptic feedback
 */

import { settings } from './settings.js';

class FeedbackManager {
  constructor() {
    this.sounds = {
      add: document.getElementById('sound-add'),
      remove: document.getElementById('sound-remove')
    };
    
    this.synth = window.speechSynthesis;
    this.voice = null;
    
    // Find a good voice
    this.loadVoices();
    if (this.synth) {
      this.synth.onvoiceschanged = () => this.loadVoices();
    }
  }

  loadVoices() {
    if (!this.synth) return;
    
    const voices = this.synth.getVoices();
    // Prefer a female English voice
    this.voice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Female')) 
              || voices.find(v => v.lang.startsWith('en'))
              || voices[0];
  }

  // Play a sound effect
  playSound(type) {
    if (!settings.get('soundEnabled')) return;
    
    const sound = this.sounds[type];
    if (sound) {
      sound.currentTime = 0;
      sound.play().catch(() => {}); // Ignore autoplay errors
    }
  }

  // Speak text
  speak(text) {
    if (!settings.get('voiceEnabled') || !this.synth) return;
    
    // Cancel any ongoing speech
    this.synth.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = this.voice;
    utterance.rate = 1.1;
    utterance.pitch = 1;
    utterance.volume = 0.8;
    
    this.synth.speak(utterance);
  }

  // Trigger haptic feedback
  vibrate(pattern = [50]) {
    if (!settings.get('hapticEnabled')) return;
    
    if (navigator.vibrate) {
      navigator.vibrate(pattern);
    }
  }

  // Combined feedback for adding item
  itemAdded(product) {
    this.playSound('add');
    this.speak(`Added ${product.name}, ${this.formatPrice(product.price)}`);
    this.vibrate([50]);
  }

  // Combined feedback for removing item
  itemRemoved(product) {
    this.playSound('remove');
    this.speak(`Removed ${product.name}`);
    this.vibrate([30, 20, 30]);
  }

  // Feedback for barcode scan
  barcodeScanned(product) {
    this.playSound('add');
    this.speak(`Scanned ${product.name}`);
    this.vibrate([50, 30, 50]); // Double pulse for barcode
  }

  // Feedback for checkout
  checkout(total) {
    this.speak(`Checkout complete. Total ${this.formatPrice(total)}`);
    this.vibrate([100, 50, 100, 50, 100]);
  }

  formatPrice(price) {
    return `$${price.toFixed(2)}`;
  }
}

export const feedback = new FeedbackManager();
export default feedback;

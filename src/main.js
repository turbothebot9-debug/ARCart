/**
 * ARCart - Main Application Entry
 * Initializes camera, ML recognition, and cart system
 */

import { ProductRecognizer } from './ml/recognizer.js';
import { CartManager } from './cart/cartManager.js';
import { CameraManager } from './utils/camera.js';
import { UIManager } from './utils/ui.js';
import { shoppingList } from './utils/shoppingList.js';

class ARCartApp {
  constructor() {
    this.camera = new CameraManager();
    this.recognizer = new ProductRecognizer();
    this.cart = new CartManager();
    this.ui = new UIManager(this.cart);
    
    this.isProcessing = false;
    this.lastDetection = null;
    this.detectionCooldown = 2000; // ms between same-product detections
  }

  async init() {
    try {
      this.ui.setStatus('Initializing camera...', 'loading');
      
      // Start camera
      await this.camera.start();
      
      this.ui.setStatus('Loading ML model...', 'loading');
      
      // Load ML model
      await this.recognizer.initialize();
      
      this.ui.setStatus('Ready - Point at products or barcodes', 'ready');
      
      // Initialize shopping list UI
      this.initShoppingListUI();
      
      // Start detection loop (ML-based)
      this.startDetectionLoop();
      
      // Start barcode scanning (runs in parallel)
      this.startBarcodeScanning();
      
      // Setup event listeners
      this.setupEventListeners();
      
    } catch (error) {
      console.error('Initialization error:', error);
      this.ui.setStatus(`Error: ${error.message}`, 'error');
    }
  }

  initShoppingListUI() {
    this.updateShoppingListDisplay();
    
    // List button toggle
    const listBtn = document.getElementById('list-btn');
    const listPanel = document.getElementById('shopping-list-panel');
    const closePanel = listPanel?.querySelector('.close-panel');
    
    listBtn?.addEventListener('click', () => {
      listPanel?.classList.toggle('hidden');
    });
    
    closePanel?.addEventListener('click', () => {
      listPanel?.classList.add('hidden');
    });
    
    // Add item form
    const addBtn = document.getElementById('add-list-item');
    const input = document.getElementById('new-list-item');
    
    const addItem = () => {
      const name = input?.value.trim();
      if (name) {
        shoppingList.addItem(name);
        input.value = '';
        this.updateShoppingListDisplay();
      }
    };
    
    addBtn?.addEventListener('click', addItem);
    input?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') addItem();
    });
    
    // Clear buttons
    document.getElementById('clear-found-btn')?.addEventListener('click', () => {
      shoppingList.clearFound();
      this.updateShoppingListDisplay();
    });
    
    document.getElementById('clear-list-btn')?.addEventListener('click', () => {
      if (confirm('Clear entire shopping list?')) {
        shoppingList.clear();
        this.updateShoppingListDisplay();
      }
    });
    
    // Update button indicator
    this.updateListButtonIndicator();
  }

  updateShoppingListDisplay() {
    const container = document.getElementById('shopping-list-items');
    if (!container) return;
    
    const items = shoppingList.getItems();
    const progress = shoppingList.getProgress();
    
    // Update progress bar
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    if (progressFill) progressFill.style.width = `${progress.percentage}%`;
    if (progressText) progressText.textContent = `${progress.found} of ${progress.total} items found`;
    
    // Update list
    if (items.length === 0) {
      container.innerHTML = `
        <div class="list-empty">
          <div class="list-empty-icon">📝</div>
          <div class="list-empty-text">Your list is empty<br>Add items you want to find</div>
        </div>
      `;
    } else {
      container.innerHTML = items.map(item => `
        <div class="list-item ${item.found ? 'found' : ''}" data-id="${item.id}">
          <div class="list-item-checkbox"></div>
          <span class="list-item-name">${this.escapeHtml(item.name)}${item.quantity > 1 ? ` (x${item.quantity})` : ''}</span>
          <button class="list-item-remove">✕</button>
        </div>
      `).join('');
      
      // Add click handlers
      container.querySelectorAll('.list-item').forEach(el => {
        const id = el.dataset.id;
        
        el.querySelector('.list-item-checkbox')?.addEventListener('click', () => {
          shoppingList.toggleFound(id);
          this.updateShoppingListDisplay();
        });
        
        el.querySelector('.list-item-remove')?.addEventListener('click', () => {
          shoppingList.removeItem(id);
          this.updateShoppingListDisplay();
        });
      });
    }
    
    this.updateListButtonIndicator();
  }

  updateListButtonIndicator() {
    const listBtn = document.getElementById('list-btn');
    if (!listBtn) return;
    
    const unfound = shoppingList.getUnfoundItems();
    if (unfound.length > 0) {
      listBtn.classList.add('has-items');
    } else {
      listBtn.classList.remove('has-items');
    }
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  startBarcodeScanning() {
    const videoElement = document.getElementById('camera-feed');
    
    this.recognizer.startBarcodeScanning(videoElement, (detection) => {
      this.handleBarcodeDetection(detection);
    });
  }

  handleBarcodeDetection(detection) {
    const { product, source, barcode } = detection;
    
    // Check if this matches something on our shopping list
    this.checkShoppingListMatch(product.name);
    
    // Add to cart
    this.cart.addItem(product);
    
    // Show popup with barcode indicator
    this.ui.showDetectionPopup(product, 'added');
    this.ui.updateCartDisplay();
    
    // Update status to show barcode was scanned
    const sourceLabel = source === 'barcode-local' ? 'Local DB' : 
                       source === 'barcode-api' ? 'Online' : 'New';
    this.ui.setStatus(`Scanned: ${barcode} (${sourceLabel})`, 'ready');
    
    // Haptic feedback
    if (navigator.vibrate) {
      navigator.vibrate([50, 30, 50]); // Double pulse for barcode
    }
    
    // Reset status after delay
    setTimeout(() => {
      this.ui.setStatus('Ready - Point at products or barcodes', 'ready');
    }, 2000);
  }

  checkShoppingListMatch(productName) {
    const matchedItem = shoppingList.matchProduct(productName);
    
    if (matchedItem && !matchedItem.found) {
      // Mark as found
      shoppingList.markFound(matchedItem.id);
      this.updateShoppingListDisplay();
      
      // Show found notification
      this.showListFoundPopup(matchedItem.name);
    }
  }

  showListFoundPopup(itemName) {
    const popup = document.getElementById('list-found-popup');
    const nameEl = popup?.querySelector('.found-item-name');
    
    if (popup && nameEl) {
      nameEl.textContent = itemName;
      popup.classList.remove('hidden');
      
      // Extra haptic for list item found
      if (navigator.vibrate) {
        navigator.vibrate([100, 50, 100]);
      }
      
      setTimeout(() => {
        popup.classList.add('hidden');
      }, 3000);
    }
  }

  startDetectionLoop() {
    const detect = async () => {
      if (!this.isProcessing && this.camera.isReady()) {
        this.isProcessing = true;
        
        try {
          const frame = this.camera.getFrame();
          const detections = await this.recognizer.detect(frame);
          
          this.processDetections(detections);
          
        } catch (error) {
          console.error('Detection error:', error);
        }
        
        this.isProcessing = false;
      }
      
      // Run at ~10 FPS for detection (balance between responsiveness and performance)
      requestAnimationFrame(() => setTimeout(detect, 100));
    };
    
    detect();
  }

  processDetections(detections) {
    // Clear previous boxes
    this.ui.clearDetectionBoxes();
    
    if (detections.length === 0) return;
    
    // Get the highest confidence detection
    const bestDetection = detections.reduce((best, current) => 
      current.confidence > best.confidence ? current : best
    );
    
    // Only process if confidence is high enough
    if (bestDetection.confidence < 0.6) return;
    
    // Draw bounding box
    this.ui.drawDetectionBox(bestDetection);
    
    // Check cooldown for same product
    const now = Date.now();
    if (this.lastDetection && 
        this.lastDetection.product.id === bestDetection.product.id &&
        now - this.lastDetection.timestamp < this.detectionCooldown) {
      return;
    }
    
    // Product detected!
    this.lastDetection = {
      product: bestDetection.product,
      timestamp: now
    };
    
    // Check shopping list match
    this.checkShoppingListMatch(bestDetection.product.name);
    
    // Add to cart and show popup
    this.cart.addItem(bestDetection.product);
    this.ui.showDetectionPopup(bestDetection.product, 'added');
    this.ui.updateCartDisplay();
    
    // Haptic feedback if available
    if (navigator.vibrate) {
      navigator.vibrate(50);
    }
  }

  setupEventListeners() {
    // Cart summary click
    document.getElementById('cart-summary').addEventListener('click', () => {
      this.ui.toggleCartDrawer();
    });
    
    // Toggle cart button
    document.getElementById('toggle-cart').addEventListener('click', () => {
      this.ui.toggleCartDrawer();
    });
    
    // Close cart
    document.getElementById('close-cart').addEventListener('click', () => {
      this.ui.hideCartDrawer();
    });
    
    // Checkout button
    document.getElementById('checkout-btn').addEventListener('click', () => {
      this.handleCheckout();
    });
  }

  handleCheckout() {
    const total = this.cart.getTotal();
    const items = this.cart.getItems();
    
    if (items.length === 0) {
      alert('Your cart is empty!');
      return;
    }
    
    // For now, just show confirmation
    const itemList = items.map(i => `${i.name} - $${i.price.toFixed(2)}`).join('\n');
    const confirmed = confirm(
      `Ready to checkout?\n\n${itemList}\n\nTotal: $${total.toFixed(2)}`
    );
    
    if (confirmed) {
      alert('✓ Payment successful!\n\nReceipt sent to your phone.');
      this.cart.clear();
      this.ui.updateCartDisplay();
      this.ui.hideCartDrawer();
    }
  }
}

// Start the app
const app = new ARCartApp();
app.init();

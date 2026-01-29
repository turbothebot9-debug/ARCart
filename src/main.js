/**
 * ARCart - Main Application Entry
 * Initializes camera, ML recognition, and cart system
 */

import { ProductRecognizer } from './ml/recognizer.js';
import { CartManager } from './cart/cartManager.js';
import { CameraManager } from './utils/camera.js';
import { UIManager } from './utils/ui.js';
import { purchaseHistory } from './utils/purchaseHistory.js';

class ARCartApp {
  constructor() {
    this.camera = new CameraManager();
    this.recognizer = new ProductRecognizer();
    this.cart = new CartManager();
    this.ui = new UIManager(this.cart);
    
    this.isProcessing = false;
    this.lastDetection = null;
    this.detectionCooldown = 2000; // ms between same-product detections
    
    this.selectedPurchaseId = null;
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
      
      // Initialize history UI
      this.initHistoryUI();
      
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

  initHistoryUI() {
    // History panel toggle
    const historyBtn = document.getElementById('history-btn');
    const historyPanel = document.getElementById('history-panel');
    const closeHistory = document.getElementById('close-history');
    
    historyBtn?.addEventListener('click', () => {
      historyPanel?.classList.toggle('hidden');
      if (!historyPanel?.classList.contains('hidden')) {
        this.updateHistoryDisplay();
      }
    });
    
    closeHistory?.addEventListener('click', () => {
      historyPanel?.classList.add('hidden');
    });
    
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`tab-${tabId}`)?.classList.add('active');
      });
    });
    
    // Receipt modal
    const receiptModal = document.getElementById('receipt-modal');
    const closeReceiptModal = document.getElementById('close-receipt-modal');
    
    closeReceiptModal?.addEventListener('click', () => {
      receiptModal?.classList.add('hidden');
    });
    
    receiptModal?.addEventListener('click', (e) => {
      if (e.target === receiptModal) {
        receiptModal.classList.add('hidden');
      }
    });
    
    // Copy receipt
    document.getElementById('copy-receipt')?.addEventListener('click', () => {
      const receiptText = document.getElementById('receipt-text')?.textContent;
      if (receiptText) {
        navigator.clipboard.writeText(receiptText).then(() => {
          alert('Receipt copied to clipboard!');
        });
      }
    });
    
    // Share receipt
    document.getElementById('share-receipt')?.addEventListener('click', async () => {
      const receiptText = document.getElementById('receipt-text')?.textContent;
      if (receiptText && navigator.share) {
        try {
          await navigator.share({
            title: 'ARCart Receipt',
            text: receiptText
          });
        } catch (e) {
          // User cancelled or error
        }
      }
    });
    
    // Initial display
    this.updateHistoryDisplay();
  }

  updateHistoryDisplay() {
    const stats = purchaseHistory.getStats();
    const history = purchaseHistory.getHistory();
    
    // Update stats
    document.getElementById('stat-total-spent').textContent = `$${stats.totalSpent.toFixed(2)}`;
    document.getElementById('stat-purchase-count').textContent = stats.purchaseCount;
    document.getElementById('stat-avg-purchase').textContent = `$${stats.averagePurchase.toFixed(2)}`;
    
    // Update recent purchases list
    const recentList = document.getElementById('recent-purchases');
    if (recentList) {
      if (history.length === 0) {
        recentList.innerHTML = `
          <div class="history-empty">
            <div class="history-empty-icon">🧾</div>
            <div class="history-empty-text">No purchases yet<br>Your receipts will appear here</div>
          </div>
        `;
      } else {
        recentList.innerHTML = history.slice(0, 10).map(purchase => {
          const date = new Date(purchase.date);
          return `
            <div class="history-item" data-id="${purchase.id}">
              <div class="history-date">
                <div class="history-day">${date.getDate()}</div>
                <div class="history-month">${date.toLocaleString('default', { month: 'short' })}</div>
              </div>
              <div class="history-details">
                <div class="history-items-count">${purchase.itemCount} item${purchase.itemCount !== 1 ? 's' : ''}</div>
                <div class="history-time">${date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}</div>
              </div>
              <div class="history-total">$${purchase.total.toFixed(2)}</div>
            </div>
          `;
        }).join('');
        
        // Add click handlers
        recentList.querySelectorAll('.history-item').forEach(item => {
          item.addEventListener('click', () => {
            this.showReceipt(item.dataset.id);
          });
        });
      }
    }
    
    // Update most bought items
    const mostBoughtList = document.getElementById('most-bought-list');
    if (mostBoughtList) {
      if (stats.mostBoughtItems.length === 0) {
        mostBoughtList.innerHTML = '<p style="color: rgba(255,255,255,0.4); font-size: 13px;">No data yet</p>';
      } else {
        mostBoughtList.innerHTML = stats.mostBoughtItems.slice(0, 5).map(item => `
          <div class="most-bought-item">
            <span class="most-bought-name">${this.escapeHtml(item.name)}</span>
            <span class="most-bought-count">${item.count}x</span>
          </div>
        `).join('');
      }
    }
    
    // Update category spending
    const categorySpending = document.getElementById('category-spending');
    if (categorySpending) {
      const categories = Object.entries(stats.spendingByCategory);
      if (categories.length === 0) {
        categorySpending.innerHTML = '<p style="color: rgba(255,255,255,0.4); font-size: 13px;">No data yet</p>';
      } else {
        const maxSpending = Math.max(...categories.map(([, v]) => v));
        categorySpending.innerHTML = categories.map(([cat, amount]) => `
          <div class="category-item">
            <span class="category-name">${cat}</span>
            <div class="category-bar">
              <div class="category-fill" style="width: ${(amount / maxSpending) * 100}%"></div>
            </div>
            <span class="category-amount">$${amount.toFixed(2)}</span>
          </div>
        `).join('');
      }
    }
  }

  showReceipt(purchaseId) {
    const purchase = purchaseHistory.getPurchase(purchaseId);
    if (!purchase) return;
    
    this.selectedPurchaseId = purchaseId;
    
    // Update receipt viewer in tab
    const receiptViewer = document.getElementById('receipt-viewer');
    if (receiptViewer) {
      receiptViewer.textContent = purchaseHistory.formatReceipt(purchase);
    }
    
    // Show receipt modal
    const receiptModal = document.getElementById('receipt-modal');
    const receiptText = document.getElementById('receipt-text');
    
    if (receiptModal && receiptText) {
      receiptText.textContent = purchaseHistory.formatReceipt(purchase);
      receiptModal.classList.remove('hidden');
    }
    
    // Switch to receipts tab
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.tab-btn[data-tab="receipts"]')?.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-receipts')?.classList.add('active');
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
    
    // Calculate total with tax
    const tax = Math.round(total * 0.0825 * 100) / 100;
    const grandTotal = Math.round((total + tax) * 100) / 100;
    
    // Show confirmation
    const itemList = items.map(i => `${i.name} - $${i.price.toFixed(2)}`).join('\n');
    const confirmed = confirm(
      `Ready to checkout?\n\n${itemList}\n\nSubtotal: $${total.toFixed(2)}\nTax: $${tax.toFixed(2)}\nTotal: $${grandTotal.toFixed(2)}`
    );
    
    if (confirmed) {
      // Record purchase in history
      const purchase = purchaseHistory.addPurchase(items, total);
      
      // Show receipt
      const receipt = purchaseHistory.formatReceipt(purchase);
      alert('✓ Payment successful!\n\n' + receipt);
      
      // Clear cart
      this.cart.clear();
      this.ui.updateCartDisplay();
      this.ui.hideCartDrawer();
      
      // Update history display if panel is open
      this.updateHistoryDisplay();
    }
  }
}

// Start the app
const app = new ARCartApp();
app.init();

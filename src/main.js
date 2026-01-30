/**
 * ARCart - Main Application Entry
 * Initializes camera, ML recognition, and cart system
 */

import { ProductRecognizer } from './ml/recognizer.js';
import { CartManager } from './cart/cartManager.js';
import { CameraManager } from './utils/camera.js';
import { UIManager } from './utils/ui.js';

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
    console.log('Detection:', bestDetection.product?.name, 'confidence:', bestDetection.confidence);
    if (bestDetection.confidence < 0.35) return;
    
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

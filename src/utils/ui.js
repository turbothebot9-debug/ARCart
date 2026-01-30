/**
 * UI Manager - Handles all UI updates and interactions
 */

export class UIManager {
  constructor(cart) {
    this.cart = cart;
    
    // Cache DOM elements
    this.elements = {
      cartCount: document.getElementById('cart-count'),
      cartTotal: document.getElementById('cart-total'),
      cartItems: document.getElementById('cart-items'),
      cartSubtotal: document.getElementById('cart-subtotal'),
      cartDrawer: document.getElementById('cart-drawer'),
      detectionPopup: document.getElementById('detection-popup'),
      detectionCanvas: document.getElementById('detection-canvas'),
      statusDot: document.getElementById('status-dot'),
      statusText: document.getElementById('status-text'),
      productImage: document.getElementById('detected-product-image'),
      productName: document.getElementById('detected-product-name'),
      productPrice: document.getElementById('detected-product-price'),
      productStatus: document.getElementById('detected-product-status')
    };
    
    this.ctx = this.elements.detectionCanvas.getContext('2d');
    this.popupTimeout = null;
    
    // Initial update
    this.updateCartDisplay();
  }

  setStatus(message, state = 'loading') {
    this.elements.statusText.textContent = message;
    this.elements.statusDot.className = '';
    
    if (state === 'ready') {
      this.elements.statusDot.classList.add('ready');
    } else if (state === 'error') {
      this.elements.statusDot.classList.add('error');
    }
  }

  updateCartDisplay() {
    const count = this.cart.getItemCount();
    const total = this.cart.getTotal();
    const items = this.cart.getItems();
    
    // Update summary
    this.elements.cartCount.textContent = count;
    this.elements.cartTotal.textContent = `$${total.toFixed(2)}`;
    this.elements.cartSubtotal.textContent = `Subtotal: $${total.toFixed(2)}`;
    
    // Update cart items list
    this.elements.cartItems.innerHTML = items.map(item => `
      <div class="cart-item" data-id="${item.id}">
        <div class="cart-item-image">${item.emoji || '📦'}</div>
        <div class="cart-item-details">
          <div class="cart-item-name">${item.name}${item.quantity > 1 ? ` (×${item.quantity})` : ''}</div>
          <div class="cart-item-price">$${(item.price * item.quantity).toFixed(2)}</div>
        </div>
        <button class="cart-item-remove" onclick="window.removeCartItem('${item.id}')">✕</button>
      </div>
    `).join('');
    
    // Expose remove function globally (quick hack - would use proper event delegation in production)
    window.removeCartItem = (id) => {
      this.cart.removeItemCompletely(id);
      this.updateCartDisplay();
    };
  }

  showDetectionPopup(product, action = 'added') {
    // Clear any existing timeout
    if (this.popupTimeout) {
      clearTimeout(this.popupTimeout);
    }
    
    // Update popup content
    this.elements.productImage.textContent = product.emoji || '📦';
    this.elements.productName.textContent = product.name;
    this.elements.productPrice.textContent = `$${product.price.toFixed(2)}`;
    
    if (action === 'added') {
      this.elements.productStatus.textContent = '✓ Added to cart';
      this.elements.productStatus.classList.remove('removed');
    } else {
      this.elements.productStatus.textContent = '✗ Removed from cart';
      this.elements.productStatus.classList.add('removed');
    }
    
    // Show popup
    this.elements.detectionPopup.classList.remove('hidden');
    
    // Auto-hide after 2 seconds
    this.popupTimeout = setTimeout(() => {
      this.elements.detectionPopup.classList.add('hidden');
    }, 2000);
  }

  hideDetectionPopup() {
    this.elements.detectionPopup.classList.add('hidden');
  }

  toggleCartDrawer() {
    this.elements.cartDrawer.classList.toggle('visible');
    this.elements.cartDrawer.classList.toggle('hidden');
  }

  showCartDrawer() {
    this.elements.cartDrawer.classList.add('visible');
    this.elements.cartDrawer.classList.remove('hidden');
  }

  hideCartDrawer() {
    this.elements.cartDrawer.classList.remove('visible');
    this.elements.cartDrawer.classList.add('hidden');
  }

  drawDetectionBox(detection) {
    const canvas = this.elements.detectionCanvas;
    const video = document.getElementById('camera-feed');
    
    // Check if we have bounding box data (category classifiers don't provide this)
    const box = detection.boundingBox || detection.bbox;
    if (!box) {
      // No bounding box - just show floating label in center-bottom
      this.showFloatingLabel(detection);
      return;
    }
    
    // Scale bounding box to canvas size
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    const x = box.x * scaleX;
    const y = box.y * scaleY;
    const width = box.width * scaleX;
    const height = box.height * scaleY;
    
    // Draw box
    this.ctx.strokeStyle = '#4CAF50';
    this.ctx.lineWidth = 3;
    this.ctx.strokeRect(x, y, width, height);
    
    // Draw label background
    const label = `${detection.product.name} - $${detection.product.price.toFixed(2)}`;
    this.ctx.font = 'bold 14px sans-serif';
    const textWidth = this.ctx.measureText(label).width;
    
    this.ctx.fillStyle = '#4CAF50';
    this.ctx.fillRect(x, y - 24, textWidth + 16, 24);
    
    // Draw label text
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText(label, x + 8, y - 7);
  }

  clearDetectionBoxes() {
    const canvas = this.elements.detectionCanvas;
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  showFloatingLabel(detection) {
    const canvas = this.elements.detectionCanvas;
    
    // Draw centered label at bottom of screen (for classifiers without bbox)
    const label = `${detection.product.name} - $${detection.product.price.toFixed(2)}`;
    const confidence = `${Math.round(detection.confidence * 100)}% confidence`;
    
    this.ctx.font = 'bold 18px sans-serif';
    const labelWidth = this.ctx.measureText(label).width;
    const x = (canvas.width - labelWidth - 24) / 2;
    const y = canvas.height - 60;
    
    // Draw pill-shaped background
    this.ctx.fillStyle = 'rgba(76, 175, 80, 0.9)';
    this.ctx.beginPath();
    this.ctx.roundRect(x, y, labelWidth + 24, 50, 12);
    this.ctx.fill();
    
    // Draw label text
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText(label, x + 12, y + 22);
    
    // Draw confidence
    this.ctx.font = '12px sans-serif';
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    this.ctx.fillText(confidence, x + 12, y + 40);
  }
}

/**
 * Cart Manager - Handles shopping cart state
 */

export class CartManager {
  constructor() {
    this.items = [];
    this.listeners = [];
    
    // Load from localStorage if available
    this.load();
  }

  addItem(product) {
    // Check if item already exists
    const existingIndex = this.items.findIndex(item => item.id === product.id);
    
    if (existingIndex >= 0) {
      // Increment quantity
      this.items[existingIndex].quantity += 1;
    } else {
      // Add new item
      this.items.push({
        ...product,
        quantity: 1,
        addedAt: Date.now()
      });
    }
    
    this.save();
    this.notifyListeners('add', product);
    
    return this.items;
  }

  removeItem(productId) {
    const index = this.items.findIndex(item => item.id === productId);
    
    if (index >= 0) {
      const removed = this.items[index];
      
      if (removed.quantity > 1) {
        // Decrease quantity
        this.items[index].quantity -= 1;
      } else {
        // Remove entirely
        this.items.splice(index, 1);
      }
      
      this.save();
      this.notifyListeners('remove', removed);
    }
    
    return this.items;
  }

  removeItemCompletely(productId) {
    const index = this.items.findIndex(item => item.id === productId);
    
    if (index >= 0) {
      const removed = this.items.splice(index, 1)[0];
      this.save();
      this.notifyListeners('remove', removed);
    }
    
    return this.items;
  }

  getItems() {
    return [...this.items];
  }

  getItemCount() {
    return this.items.reduce((total, item) => total + item.quantity, 0);
  }

  getTotal() {
    return this.items.reduce((total, item) => {
      return total + (item.price * item.quantity);
    }, 0);
  }

  clear() {
    this.items = [];
    this.save();
    this.notifyListeners('clear', null);
  }

  // Persistence
  save() {
    try {
      localStorage.setItem('arcart-items', JSON.stringify(this.items));
    } catch (e) {
      console.warn('Could not save cart to localStorage:', e);
    }
  }

  load() {
    try {
      const saved = localStorage.getItem('arcart-items');
      if (saved) {
        this.items = JSON.parse(saved);
      }
    } catch (e) {
      console.warn('Could not load cart from localStorage:', e);
      this.items = [];
    }
  }

  // Event system
  addListener(callback) {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter(l => l !== callback);
    };
  }

  notifyListeners(action, product) {
    for (const listener of this.listeners) {
      try {
        listener(action, product, this);
      } catch (e) {
        console.error('Cart listener error:', e);
      }
    }
  }

  // Analytics / History
  getHistory() {
    try {
      const history = localStorage.getItem('arcart-history');
      return history ? JSON.parse(history) : [];
    } catch (e) {
      return [];
    }
  }

  recordPurchase() {
    const purchase = {
      id: `purchase-${Date.now()}`,
      items: [...this.items],
      total: this.getTotal(),
      timestamp: Date.now()
    };

    try {
      const history = this.getHistory();
      history.push(purchase);
      localStorage.setItem('arcart-history', JSON.stringify(history));
    } catch (e) {
      console.warn('Could not save purchase history:', e);
    }

    return purchase;
  }
}

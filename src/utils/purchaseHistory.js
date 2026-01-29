/**
 * Purchase History Manager - Track past purchases and receipts
 */

const STORAGE_KEY = 'arcart_purchase_history';
const MAX_HISTORY = 50; // Keep last 50 purchases

class PurchaseHistoryManager {
  constructor() {
    this.history = this.load();
  }

  load() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  }

  save() {
    try {
      // Keep only last MAX_HISTORY entries
      if (this.history.length > MAX_HISTORY) {
        this.history = this.history.slice(-MAX_HISTORY);
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.history));
    } catch (e) {
      console.warn('Could not save purchase history:', e);
    }
  }

  // Record a new purchase
  addPurchase(items, total, paymentMethod = 'card') {
    const purchase = {
      id: Date.now().toString(),
      date: new Date().toISOString(),
      items: items.map(item => ({
        name: item.name,
        price: item.price,
        quantity: item.quantity || 1,
        category: item.category
      })),
      subtotal: total,
      tax: Math.round(total * 0.0825 * 100) / 100, // 8.25% tax example
      total: Math.round(total * 1.0825 * 100) / 100,
      paymentMethod,
      itemCount: items.reduce((sum, item) => sum + (item.quantity || 1), 0)
    };
    
    this.history.push(purchase);
    this.save();
    
    return purchase;
  }

  // Get all purchases
  getHistory() {
    return [...this.history].reverse(); // Newest first
  }

  // Get purchase by ID
  getPurchase(id) {
    return this.history.find(p => p.id === id);
  }

  // Get purchases for a specific date range
  getPurchasesInRange(startDate, endDate) {
    const start = new Date(startDate).getTime();
    const end = new Date(endDate).getTime();
    
    return this.history.filter(p => {
      const purchaseDate = new Date(p.date).getTime();
      return purchaseDate >= start && purchaseDate <= end;
    });
  }

  // Get today's purchases
  getTodaysPurchases() {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    return this.getPurchasesInRange(today, tomorrow);
  }

  // Get this week's purchases
  getThisWeeksPurchases() {
    const today = new Date();
    const weekStart = new Date(today);
    weekStart.setDate(weekStart.getDate() - weekStart.getDay());
    weekStart.setHours(0, 0, 0, 0);
    
    return this.getPurchasesInRange(weekStart, today);
  }

  // Get spending statistics
  getStats() {
    if (this.history.length === 0) {
      return {
        totalSpent: 0,
        purchaseCount: 0,
        averagePurchase: 0,
        mostBoughtItems: [],
        spendingByCategory: {}
      };
    }

    const totalSpent = this.history.reduce((sum, p) => sum + p.total, 0);
    const purchaseCount = this.history.length;
    
    // Count items
    const itemCounts = {};
    const categorySpending = {};
    
    this.history.forEach(purchase => {
      purchase.items.forEach(item => {
        // Item counts
        const key = item.name.toLowerCase();
        itemCounts[key] = (itemCounts[key] || 0) + (item.quantity || 1);
        
        // Category spending
        const cat = item.category || 'other';
        categorySpending[cat] = (categorySpending[cat] || 0) + item.price * (item.quantity || 1);
      });
    });
    
    // Get most bought items
    const mostBoughtItems = Object.entries(itemCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([name, count]) => ({ name, count }));
    
    return {
      totalSpent: Math.round(totalSpent * 100) / 100,
      purchaseCount,
      averagePurchase: Math.round((totalSpent / purchaseCount) * 100) / 100,
      mostBoughtItems,
      spendingByCategory: categorySpending
    };
  }

  // Format receipt for display
  formatReceipt(purchase) {
    const date = new Date(purchase.date);
    const dateStr = date.toLocaleDateString('en-US', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'
    });
    const timeStr = date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit'
    });
    
    let receipt = [];
    receipt.push('═══════════════════════════');
    receipt.push('         ARCart Receipt');
    receipt.push('═══════════════════════════');
    receipt.push(`Date: ${dateStr}`);
    receipt.push(`Time: ${timeStr}`);
    receipt.push(`Order #: ${purchase.id.slice(-8)}`);
    receipt.push('───────────────────────────');
    receipt.push('');
    
    purchase.items.forEach(item => {
      const qty = item.quantity > 1 ? `x${item.quantity} ` : '';
      const price = `$${(item.price * (item.quantity || 1)).toFixed(2)}`;
      const name = item.name.substring(0, 20);
      receipt.push(`${qty}${name.padEnd(20)} ${price.padStart(8)}`);
    });
    
    receipt.push('');
    receipt.push('───────────────────────────');
    receipt.push(`Subtotal:${('$' + purchase.subtotal.toFixed(2)).padStart(18)}`);
    receipt.push(`Tax:${('$' + purchase.tax.toFixed(2)).padStart(23)}`);
    receipt.push('───────────────────────────');
    receipt.push(`TOTAL:${('$' + purchase.total.toFixed(2)).padStart(21)}`);
    receipt.push('═══════════════════════════');
    receipt.push('');
    receipt.push('    Thank you for shopping!');
    receipt.push('');
    
    return receipt.join('\n');
  }

  // Export history as JSON
  exportAsJSON() {
    return JSON.stringify(this.history, null, 2);
  }

  // Clear all history
  clear() {
    this.history = [];
    this.save();
  }

  // Delete a specific purchase
  deletePurchase(id) {
    this.history = this.history.filter(p => p.id !== id);
    this.save();
  }
}

export const purchaseHistory = new PurchaseHistoryManager();
export default purchaseHistory;

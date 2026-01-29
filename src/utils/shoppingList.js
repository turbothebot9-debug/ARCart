/**
 * Shopping List Manager - Pre-plan items to find in the store
 */

const STORAGE_KEY = 'arcart_shopping_list';

class ShoppingListManager {
  constructor() {
    this.items = this.load();
    this.listeners = [];
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
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.items));
    } catch (e) {
      console.warn('Could not save shopping list:', e);
    }
    this.notify();
  }

  addItem(name, quantity = 1, category = null) {
    const id = Date.now().toString();
    const item = {
      id,
      name: name.trim(),
      quantity,
      category,
      found: false,
      addedAt: new Date().toISOString()
    };
    this.items.push(item);
    this.save();
    return item;
  }

  removeItem(id) {
    this.items = this.items.filter(item => item.id !== id);
    this.save();
  }

  toggleFound(id) {
    const item = this.items.find(item => item.id === id);
    if (item) {
      item.found = !item.found;
      this.save();
    }
    return item;
  }

  markFound(id) {
    const item = this.items.find(item => item.id === id);
    if (item) {
      item.found = true;
      item.foundAt = new Date().toISOString();
      this.save();
    }
    return item;
  }

  updateQuantity(id, quantity) {
    const item = this.items.find(item => item.id === id);
    if (item) {
      item.quantity = Math.max(1, quantity);
      this.save();
    }
    return item;
  }

  getItems() {
    return [...this.items];
  }

  getUnfoundItems() {
    return this.items.filter(item => !item.found);
  }

  getFoundItems() {
    return this.items.filter(item => item.found);
  }

  getProgress() {
    const total = this.items.length;
    const found = this.items.filter(item => item.found).length;
    return {
      total,
      found,
      remaining: total - found,
      percentage: total > 0 ? Math.round((found / total) * 100) : 0
    };
  }

  clear() {
    this.items = [];
    this.save();
  }

  clearFound() {
    this.items = this.items.filter(item => !item.found);
    this.save();
  }

  // Check if a product name matches any list item
  matchProduct(productName) {
    const searchName = productName.toLowerCase();
    
    for (const item of this.items) {
      if (item.found) continue;
      
      const listName = item.name.toLowerCase();
      
      // Check for exact match or partial match
      if (searchName.includes(listName) || listName.includes(searchName)) {
        return item;
      }
      
      // Check category match
      if (item.category && searchName.includes(item.category.toLowerCase())) {
        return item;
      }
    }
    
    return null;
  }

  onChange(callback) {
    this.listeners.push(callback);
  }

  notify() {
    this.listeners.forEach(cb => cb(this.items));
  }

  // Export list as text
  exportAsText() {
    const lines = ['Shopping List', '=============', ''];
    
    const unfound = this.getUnfoundItems();
    const found = this.getFoundItems();
    
    if (unfound.length > 0) {
      lines.push('To Find:');
      unfound.forEach(item => {
        lines.push(`  [ ] ${item.name}${item.quantity > 1 ? ` (x${item.quantity})` : ''}`);
      });
      lines.push('');
    }
    
    if (found.length > 0) {
      lines.push('Found:');
      found.forEach(item => {
        lines.push(`  [✓] ${item.name}${item.quantity > 1 ? ` (x${item.quantity})` : ''}`);
      });
    }
    
    return lines.join('\n');
  }

  // Import from text (one item per line)
  importFromText(text) {
    const lines = text.split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0 && !line.startsWith('#') && !line.startsWith('='));
    
    lines.forEach(line => {
      // Parse optional quantity like "Milk x2" or "2x Milk" or "Milk (2)"
      let name = line;
      let quantity = 1;
      
      const quantityMatch = line.match(/^(\d+)x\s+(.+)$/) 
                         || line.match(/^(.+?)\s+x(\d+)$/)
                         || line.match(/^(.+?)\s*\((\d+)\)$/);
      
      if (quantityMatch) {
        if (quantityMatch[1].match(/^\d+$/)) {
          quantity = parseInt(quantityMatch[1]);
          name = quantityMatch[2];
        } else {
          name = quantityMatch[1];
          quantity = parseInt(quantityMatch[2]);
        }
      }
      
      // Skip if already exists
      if (!this.items.find(item => item.name.toLowerCase() === name.toLowerCase())) {
        this.addItem(name, quantity);
      }
    });
  }
}

export const shoppingList = new ShoppingListManager();
export default shoppingList;

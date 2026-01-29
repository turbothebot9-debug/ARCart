/**
 * Product Database - Maps ML detections to actual products
 * This is where store inventory would connect
 */

export class ProductDatabase {
  constructor() {
    // Mock product database
    // In production, this would come from a store's API
    this.products = [
      // Beverages
      {
        id: 'water-bottle-1',
        name: 'Spring Water',
        price: 1.99,
        category: 'beverages',
        emoji: '💧',
        keywords: ['water bottle', 'bottle', 'water'],
        mlClasses: ['water bottle', 'bottle', 'plastic bottle']
      },
      {
        id: 'cola-1',
        name: 'Cola Classic',
        price: 2.49,
        category: 'beverages',
        emoji: '🥤',
        keywords: ['soda', 'cola', 'pop', 'coke'],
        mlClasses: ['bottle', 'pop bottle', 'soda bottle']
      },
      {
        id: 'coffee-1',
        name: 'Iced Coffee',
        price: 4.99,
        category: 'beverages',
        emoji: '☕',
        keywords: ['coffee', 'iced coffee', 'cold brew'],
        mlClasses: ['cup', 'coffee']
      },
      
      // Fruits
      {
        id: 'banana-1',
        name: 'Organic Banana',
        price: 0.29,
        category: 'produce',
        emoji: '🍌',
        keywords: ['banana'],
        mlClasses: ['banana']
      },
      {
        id: 'apple-1',
        name: 'Honeycrisp Apple',
        price: 1.49,
        category: 'produce',
        emoji: '🍎',
        keywords: ['apple'],
        mlClasses: ['apple', 'Granny Smith']
      },
      {
        id: 'orange-1',
        name: 'Navel Orange',
        price: 0.99,
        category: 'produce',
        emoji: '🍊',
        keywords: ['orange'],
        mlClasses: ['orange']
      },
      
      // Snacks
      {
        id: 'donut-1',
        name: 'Glazed Donut',
        price: 1.99,
        category: 'bakery',
        emoji: '🍩',
        keywords: ['donut', 'doughnut'],
        mlClasses: ['donut', 'doughnut']
      },
      {
        id: 'sandwich-1',
        name: 'Turkey Sandwich',
        price: 6.99,
        category: 'deli',
        emoji: '🥪',
        keywords: ['sandwich', 'sub'],
        mlClasses: ['sandwich', 'submarine sandwich']
      },
      {
        id: 'pizza-1',
        name: 'Pizza Slice',
        price: 3.99,
        category: 'hot food',
        emoji: '🍕',
        keywords: ['pizza'],
        mlClasses: ['pizza', 'pizza pie']
      },
      
      // Other
      {
        id: 'book-1',
        name: 'Paperback Book',
        price: 12.99,
        category: 'books',
        emoji: '📖',
        keywords: ['book', 'paperback'],
        mlClasses: ['book', 'notebook']
      },
      {
        id: 'phone-charger-1',
        name: 'Phone Charger',
        price: 14.99,
        category: 'electronics',
        emoji: '🔌',
        keywords: ['charger', 'phone', 'cable'],
        mlClasses: ['cell phone', 'cellular telephone']
      }
    ];
    
    // Build index for fast lookup
    this.classIndex = new Map();
    for (const product of this.products) {
      for (const mlClass of product.mlClasses) {
        const key = mlClass.toLowerCase();
        if (!this.classIndex.has(key)) {
          this.classIndex.set(key, []);
        }
        this.classIndex.get(key).push(product);
      }
    }
  }

  /**
   * Match a detection to a product in the database
   */
  matchProduct(cocoClass, mobileNetClassifications) {
    // First try direct COCO class match
    const cocoKey = cocoClass.toLowerCase();
    if (this.classIndex.has(cocoKey)) {
      const matches = this.classIndex.get(cocoKey);
      if (matches.length === 1) {
        return matches[0];
      }
      // Multiple matches - use MobileNet to narrow down
      return this.narrowDownMatch(matches, mobileNetClassifications);
    }
    
    // Try MobileNet classifications
    for (const classification of mobileNetClassifications) {
      const key = classification.className.toLowerCase();
      
      // Check each product for keyword match
      for (const product of this.products) {
        for (const keyword of product.keywords) {
          if (key.includes(keyword) || keyword.includes(key)) {
            return product;
          }
        }
      }
    }
    
    // No match found - return generic product
    return this.createGenericProduct(cocoClass, mobileNetClassifications);
  }

  narrowDownMatch(candidates, classifications) {
    // Score each candidate based on MobileNet results
    let bestMatch = candidates[0];
    let bestScore = 0;
    
    for (const candidate of candidates) {
      let score = 0;
      for (const classification of classifications) {
        const className = classification.className.toLowerCase();
        for (const keyword of candidate.keywords) {
          if (className.includes(keyword)) {
            score += classification.probability;
          }
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestMatch = candidate;
      }
    }
    
    return bestMatch;
  }

  createGenericProduct(cocoClass, classifications) {
    // Create a generic product entry for unknown items
    const topClass = classifications[0];
    return {
      id: `unknown-${Date.now()}`,
      name: this.formatClassName(topClass?.className || cocoClass),
      price: 0.00, // Unknown price
      category: 'unknown',
      emoji: '📦',
      isGeneric: true
    };
  }

  formatClassName(className) {
    // Convert "water_bottle" -> "Water Bottle"
    return className
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  }

  addProduct(product) {
    this.products.push(product);
    // Update index
    for (const mlClass of (product.mlClasses || [])) {
      const key = mlClass.toLowerCase();
      if (!this.classIndex.has(key)) {
        this.classIndex.set(key, []);
      }
      this.classIndex.get(key).push(product);
    }
  }

  getProduct(id) {
    return this.products.find(p => p.id === id);
  }

  getAllProducts() {
    return this.products;
  }
}

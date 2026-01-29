/**
 * Product Database - Maps ML detections to actual products
 * This is where store inventory would connect
 */

export class ProductDatabase {
  constructor() {
    // Barcode to product mapping
    // In production, this would use a UPC database API
    this.barcodeIndex = new Map();
    
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
        mlClasses: ['water bottle', 'bottle', 'plastic bottle'],
        barcodes: ['012000001234', '4052648068025']
      },
      {
        id: 'cola-1',
        name: 'Cola Classic',
        price: 2.49,
        category: 'beverages',
        emoji: '🥤',
        keywords: ['soda', 'cola', 'pop', 'coke'],
        mlClasses: ['bottle', 'pop bottle', 'soda bottle'],
        barcodes: ['049000006346', '049000028911']
      },
      {
        id: 'coffee-1',
        name: 'Iced Coffee',
        price: 4.99,
        category: 'beverages',
        emoji: '☕',
        keywords: ['coffee', 'iced coffee', 'cold brew'],
        mlClasses: ['cup', 'coffee'],
        barcodes: ['012000171765']
      },
      
      // Fruits (usually no barcodes - sold by weight)
      {
        id: 'banana-1',
        name: 'Organic Banana',
        price: 0.29,
        category: 'produce',
        emoji: '🍌',
        keywords: ['banana'],
        mlClasses: ['banana'],
        barcodes: ['4011'] // PLU code
      },
      {
        id: 'apple-1',
        name: 'Honeycrisp Apple',
        price: 1.49,
        category: 'produce',
        emoji: '🍎',
        keywords: ['apple'],
        mlClasses: ['apple', 'Granny Smith'],
        barcodes: ['3283', '4128'] // PLU codes
      },
      {
        id: 'orange-1',
        name: 'Navel Orange',
        price: 0.99,
        category: 'produce',
        emoji: '🍊',
        keywords: ['orange'],
        mlClasses: ['orange'],
        barcodes: ['3107', '4012'] // PLU codes
      },
      
      // Snacks
      {
        id: 'donut-1',
        name: 'Glazed Donut',
        price: 1.99,
        category: 'bakery',
        emoji: '🍩',
        keywords: ['donut', 'doughnut'],
        mlClasses: ['donut', 'doughnut'],
        barcodes: []
      },
      {
        id: 'sandwich-1',
        name: 'Turkey Sandwich',
        price: 6.99,
        category: 'deli',
        emoji: '🥪',
        keywords: ['sandwich', 'sub'],
        mlClasses: ['sandwich', 'submarine sandwich'],
        barcodes: []
      },
      {
        id: 'pizza-1',
        name: 'Pizza Slice',
        price: 3.99,
        category: 'hot food',
        emoji: '🍕',
        keywords: ['pizza'],
        mlClasses: ['pizza', 'pizza pie'],
        barcodes: []
      },
      
      // Packaged goods with barcodes
      {
        id: 'chips-1',
        name: "Lay's Classic Chips",
        price: 4.29,
        category: 'snacks',
        emoji: '🥔',
        keywords: ['chips', 'crisps', 'lays'],
        mlClasses: ['bag', 'packet'],
        barcodes: ['028400443159', '028400083652']
      },
      {
        id: 'cereal-1',
        name: 'Cheerios',
        price: 5.49,
        category: 'breakfast',
        emoji: '🥣',
        keywords: ['cereal', 'cheerios'],
        mlClasses: ['box', 'carton'],
        barcodes: ['016000275287', '016000487925']
      },
      {
        id: 'milk-1',
        name: 'Whole Milk (1 gal)',
        price: 3.99,
        category: 'dairy',
        emoji: '🥛',
        keywords: ['milk', 'dairy'],
        mlClasses: ['bottle', 'jug'],
        barcodes: ['041130007897', '070852993317']
      },
      
      // Other
      {
        id: 'book-1',
        name: 'Paperback Book',
        price: 12.99,
        category: 'books',
        emoji: '📖',
        keywords: ['book', 'paperback'],
        mlClasses: ['book', 'notebook'],
        barcodes: ['9780140449136'] // ISBN example
      },
      {
        id: 'phone-charger-1',
        name: 'Phone Charger',
        price: 14.99,
        category: 'electronics',
        emoji: '🔌',
        keywords: ['charger', 'phone', 'cable'],
        mlClasses: ['cell phone', 'cellular telephone'],
        barcodes: ['190199246591']
      }
    ];
    
    // Build barcode index
    this.buildBarcodeIndex();
    
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

  buildBarcodeIndex() {
    for (const product of this.products) {
      for (const barcode of (product.barcodes || [])) {
        this.barcodeIndex.set(barcode, product);
      }
    }
  }

  /**
   * Look up a product by barcode
   */
  lookupBarcode(barcode) {
    // Direct match
    if (this.barcodeIndex.has(barcode)) {
      return this.barcodeIndex.get(barcode);
    }
    
    // Try without leading zeros
    const trimmed = barcode.replace(/^0+/, '');
    if (this.barcodeIndex.has(trimmed)) {
      return this.barcodeIndex.get(trimmed);
    }
    
    // Try adding leading zero (UPC-A vs EAN-13)
    const padded = '0' + barcode;
    if (this.barcodeIndex.has(padded)) {
      return this.barcodeIndex.get(padded);
    }
    
    return null;
  }

  /**
   * Look up product via external UPC database API
   * (Fallback when not in local database)
   */
  async lookupBarcodeOnline(barcode) {
    try {
      // Using Open Food Facts API (free, no key required)
      const response = await fetch(
        `https://world.openfoodfacts.org/api/v0/product/${barcode}.json`
      );
      
      if (!response.ok) return null;
      
      const data = await response.json();
      
      if (data.status === 1 && data.product) {
        const p = data.product;
        return {
          id: `barcode-${barcode}`,
          name: p.product_name || p.product_name_en || 'Unknown Product',
          price: 0.00, // Price not available from Open Food Facts
          category: p.categories_tags?.[0]?.replace('en:', '') || 'unknown',
          emoji: '📦',
          barcode: barcode,
          brand: p.brands || '',
          image: p.image_url || null,
          isFromApi: true,
          needsPricing: true
        };
      }
      
      return null;
    } catch (error) {
      console.warn('Barcode API lookup failed:', error);
      return null;
    }
  }

  /**
   * Create unknown product from barcode
   */
  createUnknownBarcodeProduct(barcode) {
    return {
      id: `unknown-barcode-${barcode}`,
      name: `Product (${barcode})`,
      price: 0.00,
      category: 'unknown',
      emoji: '📦',
      barcode: barcode,
      isUnknown: true,
      needsPricing: true
    };
  }

  addProduct(product) {
    this.products.push(product);
    
    // Update ML class index
    for (const mlClass of (product.mlClasses || [])) {
      const key = mlClass.toLowerCase();
      if (!this.classIndex.has(key)) {
        this.classIndex.set(key, []);
      }
      this.classIndex.get(key).push(product);
    }
    
    // Update barcode index
    for (const barcode of (product.barcodes || [])) {
      this.barcodeIndex.set(barcode, product);
    }
  }

  getProduct(id) {
    return this.products.find(p => p.id === id);
  }

  getAllProducts() {
    return this.products;
  }
}

/**
 * Product Recognizer - ML-based product detection
 * Uses TensorFlow.js with MobileNet for classification
 * and COCO-SSD for object detection
 * Falls back to barcode scanning when ML is uncertain
 */

import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { ProductDatabase } from './productDatabase.js';
import { BarcodeScanner } from './barcodeScanner.js';

export class ProductRecognizer {
  constructor() {
    this.mobilenetModel = null;
    this.cocoModel = null;
    this.barcodeScanner = new BarcodeScanner();
    this.productDb = new ProductDatabase();
    this.isLoaded = false;
    this.barcodeEnabled = true;
    
    // Track recent barcode scans to avoid duplicates
    this.recentBarcodes = new Map();
    this.barcodeCooldownMs = 5000;
  }

  async loadModel() {
    console.log('Loading TensorFlow.js models...');
    
    // Load both models in parallel
    const [mobilenetLoaded, cocoLoaded] = await Promise.all([
      mobilenet.load({ version: 2, alpha: 1.0 }),
      cocoSsd.load({ base: 'mobilenet_v2' })
    ]);
    
    this.mobilenetModel = mobilenetLoaded;
    this.cocoModel = cocoLoaded;
    this.isLoaded = true;
    
    console.log('Models loaded successfully');
  }

  /**
   * Process a barcode detection
   */
  async processBarcode(barcodeData) {
    const { code, format } = barcodeData;
    
    // Check cooldown
    const now = Date.now();
    if (this.recentBarcodes.has(code)) {
      const lastScan = this.recentBarcodes.get(code);
      if (now - lastScan < this.barcodeCooldownMs) {
        return null;
      }
    }
    this.recentBarcodes.set(code, now);
    
    // Clean up old entries
    for (const [barcode, timestamp] of this.recentBarcodes) {
      if (now - timestamp > this.barcodeCooldownMs * 2) {
        this.recentBarcodes.delete(barcode);
      }
    }
    
    // Look up in local database first
    let product = this.productDb.lookupBarcode(code);
    
    if (product) {
      console.log(`Barcode ${code} matched local product: ${product.name}`);
      return {
        product: product,
        confidence: 1.0, // Barcode = exact match
        source: 'barcode-local',
        barcode: code,
        format: format
      };
    }
    
    // Try online lookup
    product = await this.productDb.lookupBarcodeOnline(code);
    
    if (product) {
      console.log(`Barcode ${code} found via API: ${product.name}`);
      // Add to local database for future lookups
      this.productDb.addProduct({
        ...product,
        barcodes: [code]
      });
      return {
        product: product,
        confidence: 0.95,
        source: 'barcode-api',
        barcode: code,
        format: format
      };
    }
    
    // Unknown barcode - create placeholder
    console.log(`Barcode ${code} not found, creating placeholder`);
    product = this.productDb.createUnknownBarcodeProduct(code);
    
    return {
      product: product,
      confidence: 0.8,
      source: 'barcode-unknown',
      barcode: code,
      format: format
    };
  }

  /**
   * Start continuous barcode scanning alongside ML detection
   */
  startBarcodeScanning(videoElement, onDetect) {
    if (!this.barcodeEnabled) return;
    
    this.barcodeScanner.start(videoElement, async (barcodeData) => {
      const result = await this.processBarcode(barcodeData);
      if (result && onDetect) {
        onDetect(result);
      }
    }).catch(err => {
      console.warn('Barcode scanner failed to start:', err);
    });
  }

  stopBarcodeScanning() {
    this.barcodeScanner.stop();
  }

  async detect(imageElement) {
    if (!this.isLoaded) {
      throw new Error('Models not loaded');
    }

    const detections = [];

    // First, use COCO-SSD to find objects in the frame
    const objects = await this.cocoModel.detect(imageElement);
    
    // Filter for relevant objects (bottles, boxes, food items, etc.)
    const relevantClasses = [
      'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 
      'sandwich', 'pizza', 'donut', 'cake', 'cell phone',
      'book', 'clock', 'vase', 'scissors', 'toothbrush'
    ];
    
    const relevantObjects = objects.filter(obj => 
      relevantClasses.includes(obj.class) && obj.score > 0.5
    );

    // For each detected object, try to classify more specifically
    for (const obj of relevantObjects) {
      // Get classification from MobileNet for more detail
      const classifications = await this.mobilenetModel.classify(imageElement, 3);
      
      // Try to match to a product in our database
      const product = this.productDb.matchProduct(obj.class, classifications);
      
      if (product) {
        detections.push({
          product: product,
          confidence: obj.score,
          boundingBox: {
            x: obj.bbox[0],
            y: obj.bbox[1],
            width: obj.bbox[2],
            height: obj.bbox[3]
          },
          rawClass: obj.class,
          classifications: classifications
        });
      }
    }

    return detections;
  }

  /**
   * For custom product training (future feature)
   * Allows users to add their own products by taking photos
   */
  async addCustomProduct(imageElement, productInfo) {
    // Get feature embedding from MobileNet
    const embedding = await this.getEmbedding(imageElement);
    
    this.productDb.addProduct({
      ...productInfo,
      embedding: embedding
    });
  }

  async getEmbedding(imageElement) {
    // Get the activation from an intermediate layer for embedding
    const img = tf.browser.fromPixels(imageElement).toFloat();
    const normalized = img.div(127.5).sub(1);
    const batched = normalized.expandDims(0);
    const resized = tf.image.resizeBilinear(batched, [224, 224]);
    
    // Get embedding (using MobileNet as feature extractor)
    const activation = this.mobilenetModel.infer(resized, true);
    const embedding = await activation.data();
    
    // Cleanup
    img.dispose();
    normalized.dispose();
    batched.dispose();
    resized.dispose();
    activation.dispose();
    
    return Array.from(embedding);
  }
}

/**
 * Product Recognizer - ML-based product detection
 * Uses TensorFlow.js with MobileNet for classification
 * and COCO-SSD for object detection
 */

import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { ProductDatabase } from './productDatabase.js';

export class ProductRecognizer {
  constructor() {
    this.mobilenetModel = null;
    this.cocoModel = null;
    this.productDb = new ProductDatabase();
    this.isLoaded = false;
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

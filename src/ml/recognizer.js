/**
 * Product Recognizer - Uses custom trained model via API
 */

import { BarcodeScanner } from './barcodeScanner.js';

// Auto-detect: if running on the server itself, use localhost; otherwise use server IP
const API_URL = window.location.hostname === '192.168.1.81' 
    ? 'http://localhost:8000' 
    : 'http://192.168.1.81:8000';

export class ProductRecognizer {
    constructor() {
        this.ready = false;
        this.classes = [];
        this.barcodeScanner = new BarcodeScanner();
    }

    async initialize() {
        try {
            // Check API health
            const healthRes = await fetch(`${API_URL}/health`, { 
                signal: AbortSignal.timeout(3000) 
            });
            if (!healthRes.ok) throw new Error('API not available');
            
            // Get class names
            const classesRes = await fetch(`${API_URL}/classes`);
            const data = await classesRes.json();
            this.classes = data.classes;
            
            this.ready = true;
            console.log(`ProductRecognizer ready with ${this.classes.length} classes`);
            return true;
        } catch (error) {
            console.warn('Custom model API not available, running in barcode-only mode:', error.message);
            this.ready = false;
            return true; // Still allow app to run with barcodes only
        }
    }

    // Alias for compatibility
    async loadModel() {
        return this.initialize();
    }

    async detect(videoElement) {
        if (!this.ready) {
            return []; // No ML detection without API
        }

        try {
            const result = await this.recognize(videoElement);
            
            if (result.confidence > 0.3) {
                return [{
                    product: {
                        id: result.category,
                        name: result.name,
                        price: this.estimatePrice(result.category),
                        category: result.category
                    },
                    confidence: result.confidence,
                    bbox: null // No bounding box from category classifier
                }];
            }
            
            return [];
        } catch (error) {
            console.error('Detection error:', error);
            return [];
        }
    }

    async recognize(imageSource) {
        if (!this.ready) {
            return this.fallbackRecognize();
        }

        try {
            // Convert image source to base64
            const base64Image = await this.imageToBase64(imageSource);
            
            // Call API
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image }),
                signal: AbortSignal.timeout(5000)
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const result = await response.json();
            
            return {
                name: result.predicted_class || 'Unknown',
                confidence: result.confidence,
                category: result.top_predictions?.[0]?.category || 'unknown',
                topPredictions: (result.top_predictions || []).map(p => ({
                    name: p.label,
                    confidence: p.score,
                    category: p.category
                }))
            };
        } catch (error) {
            console.error('Recognition error:', error);
            return this.fallbackRecognize();
        }
    }

    startBarcodeScanning(videoElement, onDetection) {
        this.barcodeScanner.start(videoElement, onDetection);
    }

    stopBarcodeScanning() {
        this.barcodeScanner.stop();
    }

    async imageToBase64(imageSource) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            if (imageSource instanceof HTMLVideoElement) {
                canvas.width = imageSource.videoWidth || 640;
                canvas.height = imageSource.videoHeight || 480;
                ctx.drawImage(imageSource, 0, 0);
                resolve(canvas.toDataURL('image/jpeg', 0.8));
                return;
            } else if (imageSource instanceof HTMLImageElement) {
                canvas.width = imageSource.naturalWidth;
                canvas.height = imageSource.naturalHeight;
                ctx.drawImage(imageSource, 0, 0);
                resolve(canvas.toDataURL('image/jpeg', 0.8));
                return;
            } else if (imageSource instanceof HTMLCanvasElement) {
                resolve(imageSource.toDataURL('image/jpeg', 0.8));
                return;
            } else if (typeof imageSource === 'string') {
                if (imageSource.startsWith('data:')) {
                    resolve(imageSource);
                    return;
                }
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => {
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    ctx.drawImage(img, 0, 0);
                    resolve(canvas.toDataURL('image/jpeg', 0.8));
                };
                img.onerror = reject;
                img.src = imageSource;
                return;
            }
            
            reject(new Error('Unsupported image source'));
        });
    }

    formatCategoryName(category) {
        return category
            .split('-')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ')
            .replace(/s$/, '');
    }

    estimatePrice(category) {
        // Rough price estimates by category
        const prices = {
            'beers': 8.99,
            'beverages': 2.49,
            'breads': 3.49,
            'candies': 1.99,
            'cereals': 4.99,
            'cheeses': 5.99,
            'chips': 3.99,
            'chocolates': 2.99,
            'coffee': 7.99,
            'condiments': 3.49,
            'cookies': 3.99,
            'crackers': 3.49,
            'dairy': 4.49,
            'energy-drinks': 2.99,
            'frozen-foods': 5.99,
            'ice-creams': 4.99,
            'juices': 3.99,
            'milks': 3.99,
            'pasta': 1.99,
            'sauces': 3.49,
            'snacks': 2.99,
            'sodas': 1.99,
            'soups': 2.49,
            'tea': 4.99,
            'waters': 1.49,
            'yogurts': 1.29
        };
        return prices[category] || 2.99;
    }

    fallbackRecognize() {
        return {
            name: 'Unknown Product',
            confidence: 0,
            category: 'unknown',
            topPredictions: []
        };
    }

    isReady() {
        return this.ready;
    }

    getClasses() {
        return this.classes;
    }
}

export default ProductRecognizer;

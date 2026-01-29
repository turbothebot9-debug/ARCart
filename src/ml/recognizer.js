/**
 * Product Recognizer - Uses custom trained model via API
 */

const API_URL = 'http://localhost:8000';

class ProductRecognizer {
    constructor() {
        this.ready = false;
        this.classes = [];
    }

    async initialize() {
        try {
            // Check API health
            const healthRes = await fetch(`${API_URL}/health`);
            if (!healthRes.ok) throw new Error('API not available');
            
            // Get class names
            const classesRes = await fetch(`${API_URL}/classes`);
            const data = await classesRes.json();
            this.classes = data.classes;
            
            this.ready = true;
            console.log(`ProductRecognizer ready with ${this.classes.length} classes`);
            return true;
        } catch (error) {
            console.warn('Custom model API not available, falling back to basic mode:', error.message);
            this.ready = false;
            return false;
        }
    }

    async recognize(imageSource) {
        if (!this.ready) {
            return this.fallbackRecognize(imageSource);
        }

        try {
            // Convert image source to base64
            const base64Image = await this.imageToBase64(imageSource);
            
            // Call API
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const result = await response.json();
            
            return {
                name: this.formatCategoryName(result.category),
                confidence: result.confidence,
                category: result.category,
                topPredictions: result.top_predictions.map(p => ({
                    name: this.formatCategoryName(p.category),
                    confidence: p.confidence
                }))
            };
        } catch (error) {
            console.error('Recognition error:', error);
            return this.fallbackRecognize(imageSource);
        }
    }

    async imageToBase64(imageSource) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Handle different image sources
            let img;
            if (imageSource instanceof HTMLVideoElement) {
                canvas.width = imageSource.videoWidth;
                canvas.height = imageSource.videoHeight;
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
                // Already base64 or URL
                if (imageSource.startsWith('data:')) {
                    resolve(imageSource);
                    return;
                }
                // Load from URL
                img = new Image();
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
        // Convert "ice-creams" to "Ice Cream"
        return category
            .split('-')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ')
            .replace(/s$/, ''); // Remove trailing 's' for singular
    }

    fallbackRecognize(imageSource) {
        // Basic fallback when API is not available
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

// Export singleton instance
export const recognizer = new ProductRecognizer();
export default recognizer;

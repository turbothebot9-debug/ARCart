/**
 * Barcode Scanner - Fallback for when ML recognition fails
 * Uses QuaggaJS for real-time barcode detection
 */

import Quagga from 'quagga';

export class BarcodeScanner {
  constructor() {
    this.isRunning = false;
    this.lastScannedCode = null;
    this.cooldownMs = 3000; // Prevent rapid re-scans
    this.lastScanTime = 0;
    this.onDetect = null;
  }

  async start(videoElement, onDetect) {
    this.onDetect = onDetect;
    
    return new Promise((resolve, reject) => {
      Quagga.init({
        inputStream: {
          name: "Live",
          type: "LiveStream",
          target: videoElement.parentElement,
          constraints: {
            facingMode: "environment",
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        },
        locator: {
          patchSize: "medium",
          halfSample: true
        },
        numOfWorkers: navigator.hardwareConcurrency || 4,
        decoder: {
          readers: [
            "ean_reader",        // EAN-13 (most common retail)
            "ean_8_reader",      // EAN-8
            "upc_reader",        // UPC-A
            "upc_e_reader",      // UPC-E
            "code_128_reader",   // Code 128
            "code_39_reader",    // Code 39
            "qr_code_reader"     // QR codes
          ]
        },
        locate: true
      }, (err) => {
        if (err) {
          console.error('Quagga init error:', err);
          reject(err);
          return;
        }
        
        Quagga.start();
        this.isRunning = true;
        this.setupDetectionHandler();
        resolve();
      });
    });
  }

  setupDetectionHandler() {
    Quagga.onDetected((result) => {
      const code = result.codeResult.code;
      const format = result.codeResult.format;
      const now = Date.now();
      
      // Cooldown check
      if (code === this.lastScannedCode && 
          now - this.lastScanTime < this.cooldownMs) {
        return;
      }
      
      // Validate barcode (basic check)
      if (!this.isValidBarcode(code, format)) {
        return;
      }
      
      this.lastScannedCode = code;
      this.lastScanTime = now;
      
      console.log(`Barcode detected: ${code} (${format})`);
      
      if (this.onDetect) {
        this.onDetect({
          code: code,
          format: format,
          timestamp: now
        });
      }
    });

    // Draw detection area (optional visual feedback)
    Quagga.onProcessed((result) => {
      if (!result) return;
      
      const canvas = Quagga.canvas.dom.overlay;
      const ctx = canvas.getContext('2d');
      
      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      if (result.boxes) {
        // Draw detected barcode regions
        result.boxes.filter(box => box !== result.box).forEach(box => {
          this.drawBox(ctx, box, '#00ff00', 2);
        });
      }
      
      if (result.box) {
        // Highlight the successful detection
        this.drawBox(ctx, result.box, '#4CAF50', 3);
      }
      
      if (result.codeResult && result.codeResult.code) {
        // Draw the barcode value
        ctx.font = 'bold 16px sans-serif';
        ctx.fillStyle = '#4CAF50';
        ctx.fillText(result.codeResult.code, result.box[0][0], result.box[0][1] - 10);
      }
    });
  }

  drawBox(ctx, box, color, lineWidth) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(box[0][0], box[0][1]);
    ctx.lineTo(box[1][0], box[1][1]);
    ctx.lineTo(box[2][0], box[2][1]);
    ctx.lineTo(box[3][0], box[3][1]);
    ctx.closePath();
    ctx.stroke();
  }

  isValidBarcode(code, format) {
    if (!code || code.length < 4) return false;
    
    // Basic format validation
    switch (format) {
      case 'ean_13':
        return /^\d{13}$/.test(code);
      case 'ean_8':
        return /^\d{8}$/.test(code);
      case 'upc_a':
        return /^\d{12}$/.test(code);
      case 'upc_e':
        return /^\d{6,8}$/.test(code);
      default:
        return code.length >= 4;
    }
  }

  stop() {
    if (this.isRunning) {
      Quagga.stop();
      this.isRunning = false;
    }
  }

  // Manual single-frame scan (for when ML fails)
  async scanFrame(imageData) {
    return new Promise((resolve) => {
      Quagga.decodeSingle({
        src: imageData,
        numOfWorkers: 0, // Use main thread for single scan
        decoder: {
          readers: [
            "ean_reader",
            "ean_8_reader", 
            "upc_reader",
            "upc_e_reader",
            "code_128_reader"
          ]
        },
        locate: true
      }, (result) => {
        if (result && result.codeResult) {
          resolve({
            code: result.codeResult.code,
            format: result.codeResult.format
          });
        } else {
          resolve(null);
        }
      });
    });
  }
}

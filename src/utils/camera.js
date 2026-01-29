/**
 * Camera Manager - Handles camera access and frame capture
 */

export class CameraManager {
  constructor() {
    this.video = document.getElementById('camera-feed');
    this.canvas = document.getElementById('detection-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.stream = null;
    this.ready = false;
  }

  async start() {
    try {
      // Request camera access - prefer back camera on mobile
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Back camera
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      this.video.srcObject = this.stream;
      
      // Wait for video to be ready
      await new Promise((resolve, reject) => {
        this.video.onloadedmetadata = () => {
          this.video.play();
          resolve();
        };
        this.video.onerror = reject;
      });

      // Set canvas size to match video
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      
      this.ready = true;
      console.log(`Camera started: ${this.video.videoWidth}x${this.video.videoHeight}`);
      
      return true;

    } catch (error) {
      console.error('Camera error:', error);
      
      if (error.name === 'NotAllowedError') {
        throw new Error('Camera access denied. Please allow camera access and reload.');
      } else if (error.name === 'NotFoundError') {
        throw new Error('No camera found on this device.');
      } else {
        throw new Error(`Camera error: ${error.message}`);
      }
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.ready = false;
  }

  isReady() {
    return this.ready && this.video.readyState >= 2;
  }

  getFrame() {
    // Return the video element directly for TensorFlow.js
    return this.video;
  }

  // Capture a still frame as ImageData
  captureFrame() {
    if (!this.isReady()) return null;
    
    this.ctx.drawImage(this.video, 0, 0);
    return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }

  // Capture as data URL (for saving/sharing)
  captureAsDataURL(type = 'image/jpeg', quality = 0.8) {
    if (!this.isReady()) return null;
    
    this.ctx.drawImage(this.video, 0, 0);
    return this.canvas.toDataURL(type, quality);
  }

  // Get video dimensions
  getDimensions() {
    return {
      width: this.video.videoWidth,
      height: this.video.videoHeight
    };
  }

  // Switch between front and back camera
  async switchCamera() {
    const currentFacing = this.stream
      ?.getVideoTracks()[0]
      ?.getSettings()
      ?.facingMode;
    
    const newFacing = currentFacing === 'environment' ? 'user' : 'environment';
    
    this.stop();
    
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: newFacing,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    
    this.video.srcObject = this.stream;
    await this.video.play();
    
    return newFacing;
  }
}

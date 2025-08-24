// Improved Emotion Detection Dashboard JavaScript

class ImprovedEmotionDashboard {
    constructor() {
        this.socket = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.isDetecting = false;
        this.fpsCounter = 0;
        this.lastFpsTime = 0;
        this.emotionChart = null;
        this.detectionHistory = [];
        
        this.initializeElements();
        this.initializeSocket();
        this.initializeWebcam();
        this.initializeChart();
        this.bindEvents();
        this.loadStats();
    }
    
    initializeElements() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('overlay');
        this.ctx = this.canvas.getContext('2d');
        
        // Buttons
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.fileInput = document.getElementById('fileInput');
        
        // Status elements
        this.detectionStatus = document.getElementById('detection-status');
        this.fpsCounter = document.getElementById('fps-counter');
        this.faceCounter = document.getElementById('face-counter');
        this.emotionCounter = document.getElementById('emotion-counter');
        
        // Results and stats
        this.detectionResults = document.getElementById('detection-results');
        this.totalDetections = document.getElementById('total-detections');
        this.totalFaces = document.getElementById('total-faces');
        this.lastDetection = document.getElementById('last-detection');
        
        // Toast
        this.toast = new bootstrap.Toast(document.getElementById('notificationToast'));
        this.toastMessage = document.getElementById('toastMessage');
    }
    
    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('🔌 Connected to server');
            this.updateDetectionStatus('🔄 Connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('🔌 Disconnected from server');
            this.updateDetectionStatus('❌ Disconnected');
        });
        
        this.socket.on('detection_results', (data) => {
            this.handleDetectionResults(data);
        });
        
        this.socket.on('stats_update', (data) => {
            this.updateStats(data);
        });
        
        this.socket.on('connection_status', (data) => {
            console.log('Connection status:', data);
        });
    }
    
    async initializeWebcam() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            this.video.play();
            
            // Set canvas size to match video
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            });
            
            console.log('✅ Webcam initialized');
            this.showNotification('Webcam initialized successfully', 'success');
            
        } catch (error) {
            console.error('❌ Error accessing webcam:', error);
            this.showNotification('Error accessing webcam: ' + error.message, 'error');
        }
    }
    
    initializeChart() {
        const ctx = document.getElementById('emotionChart').getContext('2d');
        
        this.emotionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Happy', 'Sad', 'Angry', 'Surprised', 'Fear', 'Disgust', 'Neutral'],
                datasets: [{
                    data: [0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: [
                        '#28a745', '#dc3545', '#fd7e14', '#ffc107', 
                        '#6f42c1', '#e83e8c', '#6c757d'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                return `${label}: ${value}`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    bindEvents() {
        // Start detection
        this.startBtn.addEventListener('click', () => {
            this.startDetection();
        });
        
        // Stop detection
        this.stopBtn.addEventListener('click', () => {
            this.stopDetection();
        });
        
        // Reset stats
        this.resetBtn.addEventListener('click', () => {
            this.resetStats();
        });
        
        // Capture photo
        this.captureBtn.addEventListener('click', () => {
            this.capturePhoto();
        });
        
        // Upload image
        this.uploadBtn.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // File input change
        this.fileInput.addEventListener('change', (event) => {
            this.handleFileUpload(event.target.files[0]);
        });
    }
    
    async startDetection() {
        try {
            const response = await fetch('/api/improved/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isDetecting = true;
                this.startBtn.disabled = true;
                this.stopBtn.disabled = false;
                this.updateDetectionStatus('🟢 Active');
                this.showNotification('Detection started', 'success');
                
                // Start frame sending
                this.startFrameSending();
                
            } else {
                this.showNotification('Failed to start detection: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('❌ Error starting detection:', error);
            this.showNotification('Error starting detection: ' + error.message, 'error');
        }
    }
    
    async stopDetection() {
        try {
            const response = await fetch('/api/improved/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isDetecting = false;
                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                this.updateDetectionStatus('🔴 Stopped');
                this.showNotification('Detection stopped', 'info');
                
                // Stop frame sending
                this.stopFrameSending();
                
            } else {
                this.showNotification('Failed to stop detection: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('❌ Error stopping detection:', error);
            this.showNotification('Error stopping detection: ' + error.message, 'error');
        }
    }
    
    startFrameSending() {
        this.frameInterval = setInterval(() => {
            if (this.isDetecting && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                this.sendFrame();
                this.updateFPS();
            }
        }, 100); // 10 FPS
    }
    
    stopFrameSending() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
    }
    
    sendFrame() {
        try {
            // Create canvas to capture frame
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            tempCanvas.width = this.video.videoWidth;
            tempCanvas.height = this.video.videoHeight;
            
            // Draw video frame to canvas
            tempCtx.drawImage(this.video, 0, 0);
            
            // Convert to base64
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            
            // Send via Socket.IO
            this.socket.emit('frame_data', { image_data: imageData });
            
        } catch (error) {
            console.error('❌ Error sending frame:', error);
        }
    }
    
    updateFPS() {
        const now = performance.now();
        this.fpsCounter++;
        
        if (now - this.lastFpsTime >= 1000) {
            document.getElementById('fps-counter').textContent = `FPS: ${this.fpsCounter}`;
            this.fpsCounter = 0;
            this.lastFpsTime = now;
        }
    }
    
    handleDetectionResults(data) {
        console.log('📊 Detection results:', data);
        
        // Update counters
        this.faceCounter.textContent = `Faces: ${data.faces_detected || 0}`;
        this.emotionCounter.textContent = `Emotions: ${data.emotions ? data.emotions.length : 0}`;
        
        // Update detection results display
        this.updateDetectionResultsDisplay(data);
        
        // Update chart
        this.updateEmotionChart(data);
        
        // Store in history
        if (data.emotions && data.emotions.length > 0) {
            this.detectionHistory.push({
                timestamp: data.timestamp,
                emotions: data.emotions
            });
            
            // Keep only last 50 detections
            if (this.detectionHistory.length > 50) {
                this.detectionHistory = this.detectionHistory.slice(-50);
            }
        }
        
        // Draw bounding boxes
        this.drawBoundingBoxes(data.emotions || []);
    }
    
    updateDetectionResultsDisplay(data) {
        if (!data.emotions || data.emotions.length === 0) {
            this.detectionResults.innerHTML = '<p class="text-muted text-center">No faces detected</p>';
            return;
        }
        
        let html = '';
        data.emotions.forEach((emotion, index) => {
            const confidence = (emotion.emotion_confidence * 100).toFixed(1);
            const method = emotion.analysis_method || 'Unknown';
            
            html += `
                <div class="detection-item success">
                    <div class="row">
                        <div class="col-md-6">
                            <strong>Face ${index + 1}</strong><br>
                            <span class="badge bg-primary">${emotion.emotion}</span>
                            <span class="badge bg-secondary">${confidence}%</span>
                        </div>
                        <div class="col-md-6">
                            <small class="text-muted">
                                Method: ${method}<br>
                                BBox: [${emotion.bbox.join(', ')}]
                            </small>
                        </div>
                    </div>
                </div>
            `;
        });
        
        this.detectionResults.innerHTML = html;
    }
    
    updateEmotionChart(data) {
        if (!data.emotions || data.emotions.length === 0) return;
        
        // Count emotions
        const emotionCounts = {
            'Happy': 0, 'Sad': 0, 'Angry': 0, 'Surprised': 0,
            'Fear': 0, 'Disgust': 0, 'Neutral': 0
        };
        
        data.emotions.forEach(emotion => {
            const emotionName = emotion.emotion;
            if (emotionCounts.hasOwnProperty(emotionName)) {
                emotionCounts[emotionName]++;
            }
        });
        
        // Update chart data
        this.emotionChart.data.datasets[0].data = [
            emotionCounts['Happy'], emotionCounts['Sad'], emotionCounts['Angry'],
            emotionCounts['Surprised'], emotionCounts['Fear'], emotionCounts['Disgust'],
            emotionCounts['Neutral']
        ];
        
        this.emotionChart.update();
    }
    
    drawBoundingBoxes(emotions) {
        if (!emotions || emotions.length === 0) return;
        
        // Clear previous drawings
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        emotions.forEach(emotion => {
            const bbox = emotion.bbox;
            if (bbox && bbox.length === 4) {
                const [x, y, w, h] = bbox;
                
                // Scale coordinates to canvas size
                const scaleX = this.canvas.width / this.video.videoWidth;
                const scaleY = this.canvas.height / this.video.videoHeight;
                
                const scaledX = x * scaleX;
                const scaledY = y * scaleY;
                const scaledW = w * scaleX;
                const scaledH = h * scaleY;
                
                // Draw bounding box
                this.ctx.strokeStyle = '#00ff00';
                this.ctx.lineWidth = 3;
                this.ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
                
                // Draw label background
                const label = `${emotion.emotion} (${(emotion.emotion_confidence * 100).toFixed(1)}%)`;
                const labelWidth = this.ctx.measureText(label).width;
                
                this.ctx.fillStyle = '#00ff00';
                this.ctx.fillRect(scaledX, scaledY - 25, labelWidth + 10, 25);
                
                // Draw label text
                this.ctx.fillStyle = '#000';
                this.ctx.font = '14px Arial';
                this.ctx.fillText(label, scaledX + 5, scaledY - 8);
            }
        });
    }
    
    async capturePhoto() {
        try {
            // Create canvas to capture frame
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            tempCanvas.width = this.video.videoWidth;
            tempCanvas.height = this.video.videoHeight;
            
            // Draw video frame to canvas
            tempCtx.drawImage(this.video, 0, 0);
            
            // Convert to base64
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server for processing
            const response = await fetch('/api/improved/capture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_data: imageData })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showNotification('Capture error: ' + result.error, 'error');
            } else {
                this.showNotification('Photo captured and processed!', 'success');
                this.handleDetectionResults(result);
            }
            
        } catch (error) {
            console.error('❌ Error capturing photo:', error);
            this.showNotification('Error capturing photo: ' + error.message, 'error');
        }
    }
    
    async handleFileUpload(file) {
        if (!file) return;
        
        try {
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch('/api/improved/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showNotification('Upload error: ' + result.error, 'error');
            } else {
                this.showNotification('Image uploaded and processed!', 'success');
                this.handleDetectionResults(result);
            }
            
        } catch (error) {
            console.error('❌ Error uploading file:', error);
            this.showNotification('Error uploading file: ' + error.message, 'error');
        }
    }
    
    async resetStats() {
        try {
            const response = await fetch('/api/improved/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('Statistics reset successfully', 'success');
                this.loadStats();
                this.detectionHistory = [];
                this.updateEmotionChart({ emotions: [] });
            } else {
                this.showNotification('Failed to reset stats: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('❌ Error resetting stats:', error);
            this.showNotification('Error resetting stats: ' + error.message, 'error');
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch('/api/improved/stats');
            const stats = await response.json();
            
            if (stats.error) {
                console.error('❌ Error loading stats:', stats.error);
                return;
            }
            
            this.updateStats(stats);
            
        } catch (error) {
            console.error('❌ Error loading stats:', error);
        }
    }
    
    updateStats(stats) {
        if (stats.error) {
            console.error('❌ Stats error:', stats.error);
            return;
        }
        
        this.totalDetections.textContent = stats.total_detections || 0;
        this.totalFaces.textContent = stats.total_faces || 0;
        this.lastDetection.textContent = stats.last_detection || 'Never';
        
        // Update emotion chart with historical data
        if (stats.emotion_distribution) {
            const chartData = [
                stats.emotion_distribution['Happy'] || 0,
                stats.emotion_distribution['Sad'] || 0,
                stats.emotion_distribution['Angry'] || 0,
                stats.emotion_distribution['Surprised'] || 0,
                stats.emotion_distribution['Fear'] || 0,
                stats.emotion_distribution['Disgust'] || 0,
                stats.emotion_distribution['Neutral'] || 0
            ];
            
            this.emotionChart.data.datasets[0].data = chartData;
            this.emotionChart.update();
        }
    }
    
    updateDetectionStatus(status) {
        this.detectionStatus.textContent = status;
    }
    
    showNotification(message, type = 'info') {
        this.toastMessage.textContent = message;
        
        // Update toast appearance based on type
        const toastElement = document.getElementById('notificationToast');
        toastElement.className = `toast ${type}-state`;
        
        this.toast.show();
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Improved Emotion Detection Dashboard...');
    window.dashboard = new ImprovedEmotionDashboard();
}); 
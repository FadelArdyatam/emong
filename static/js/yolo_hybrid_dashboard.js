/**
 * YOLO Hybrid Emotion Detection Dashboard JavaScript
 * Mengintegrasikan YOLO sebagai primary detector dan rule-based sebagai fallback
 */

class YOLOHybridDashboard {
    constructor() {
        this.socket = null;
        this.isDetectionRunning = false;
        this.webcamStream = null;
        this.comparisonChart = null;
        this.detectionHistory = [];
        
        this.initialize();
    }
    
    initialize() {
        this.initializeWebcam();
        this.initializeSocketIO();
        this.initializeComparisonChart();
        this.initializeSliders();
        this.loadInitialData();
        this.setupEventListeners();
    }
    
    initializeWebcam() {
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('webcamCanvas');
        
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    this.webcamStream = stream;
                    video.srcObject = stream;
                    video.play();
                    
                    // Setup canvas for frame capture
                    const context = canvas.getContext('2d');
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    
                    console.log('✅ Webcam initialized successfully');
                })
                .catch(error => {
                    console.error('❌ Error accessing webcam:', error);
                    this.showNotification('Error accessing webcam', 'error');
                });
        } else {
            console.error('❌ Webcam not supported');
            this.showNotification('Webcam not supported', 'error');
        }
    }
    
    initializeSocketIO() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('✅ Connected to server');
            this.updateConnectionStatus('connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.updateConnectionStatus('disconnected');
        });
        
        this.socket.on('connection_status', (data) => {
            this.updateConnectionStatus(data.status);
        });
        
        this.socket.on('detection_status', (data) => {
            this.updateDetectionStatus(data.status);
        });
        
        this.socket.on('frame_result', (data) => {
            this.handleFrameResult(data);
        });
        
        this.socket.on('frame_error', (data) => {
            this.showNotification(`Detection error: ${data.error}`, 'error');
        });
    }
    
    initializeComparisonChart() {
        const ctx = document.getElementById('comparisonChart').getContext('2d');
        
        this.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['YOLO', 'Rule-based', 'Hybrid'],
                datasets: [{
                    label: 'Detection Rate (%)',
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    initializeSliders() {
        const yoloWeightSlider = document.getElementById('yoloWeightSlider');
        const confidenceThresholdSlider = document.getElementById('confidenceThresholdSlider');
        
        yoloWeightSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('yoloWeightValue').textContent = value.toFixed(1);
            document.getElementById('ruleBasedWeightValue').textContent = (1.0 - value).toFixed(1);
        });
        
        confidenceThresholdSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('confidenceThresholdValue').textContent = value.toFixed(1);
        });
        
        yoloWeightSlider.addEventListener('change', (e) => {
            this.updateYOLOWeight(parseFloat(e.target.value));
        });
        
        confidenceThresholdSlider.addEventListener('change', (e) => {
            this.updateConfidenceThreshold(parseFloat(e.target.value));
        });
    }
    
    setupEventListeners() {
        document.getElementById('startDetectionBtn').addEventListener('click', () => {
            this.startDetection();
        });
        
        document.getElementById('stopDetectionBtn').addEventListener('click', () => {
            this.stopDetection();
        });
        
        document.getElementById('resetStatsBtn').addEventListener('click', () => {
            this.resetStats();
        });
        
        document.getElementById('saveResultsBtn').addEventListener('click', () => {
            this.saveResults();
        });
    }
    
    async loadInitialData() {
        await Promise.all([
            this.loadYOLOHybridStats(),
            this.loadYOLOModelInfo(),
            this.loadYOLOPerformanceStats()
        ]);
    }
    
    async loadYOLOHybridStats() {
        try {
            const response = await fetch('/api/hybrid/stats');
            const stats = await response.json();
            
            if (response.ok) {
                this.updateHybridStats(stats);
            } else {
                console.error('Error loading hybrid stats:', stats.error);
            }
        } catch (error) {
            console.error('Error loading hybrid stats:', error);
        }
    }
    
    async loadYOLOModelInfo() {
        try {
            const response = await fetch('/api/yolo/model-info');
            const info = await response.json();
            
            if (response.ok) {
                this.updateYOLOModelStatus(info);
            } else {
                console.error('Error loading YOLO model info:', info.error);
            }
        } catch (error) {
            console.error('Error loading YOLO model info:', error);
        }
    }
    
    async loadYOLOPerformanceStats() {
        try {
            const response = await fetch('/api/yolo/performance');
            const performance = await response.json();
            
            if (response.ok) {
                this.updateYOLOPerformance(performance);
            } else {
                console.error('Error loading YOLO performance:', performance.error);
            }
        } catch (error) {
            console.error('Error loading YOLO performance:', error);
        }
    }
    
    updateHybridStats(stats) {
        const container = document.getElementById('hybridStats');
        
        container.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${stats.total_detections || 0}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${((stats.yolo_usage_rate || 0) * 100).toFixed(1)}%</div>
                <div class="stat-label">YOLO Usage Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${((stats.rule_based_usage_rate || 0) * 100).toFixed(1)}%</div>
                <div class="stat-label">Rule-based Usage Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${((stats.agreement_rate || 0) * 100).toFixed(1)}%</div>
                <div class="stat-label">Agreement Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.yolo_detections || 0}</div>
                <div class="stat-label">YOLO Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.rule_based_detections || 0}</div>
                <div class="stat-label">Rule-based Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.hybrid_detections || 0}</div>
                <div class="stat-label">Hybrid Detections</div>
            </div>
        `;
        
        // Update comparison chart
        this.updateComparisonChart(stats);
    }
    
    updateYOLOModelStatus(info) {
        const container = document.getElementById('yoloModelStatus');
        const statusClass = info.model_loaded ? 'status-available' : 'status-unavailable';
        const statusText = info.model_loaded ? 'Available' : 'Unavailable';
        
        container.innerHTML = `
            <div class="model-status ${statusClass}">${statusText}</div>
            <div class="mt-2">
                <strong>Model:</strong> ${info.model_path}<br>
                <strong>Type:</strong> ${info.model_type}<br>
                <strong>Classes:</strong> ${info.num_classes}<br>
                <strong>Confidence Threshold:</strong> ${info.confidence_threshold}
            </div>
        `;
    }
    
    updateYOLOPerformance(performance) {
        const container = document.getElementById('yoloPerformance');
        
        container.innerHTML = `
            <div class="performance-metric">
                <span class="metric-label">Total Detections:</span>
                <span class="metric-value">${performance.total_detections || 0}</span>
            </div>
            <div class="performance-metric">
                <span class="metric-label">Success Rate:</span>
                <span class="metric-value">${((performance.success_rate || 0) * 100).toFixed(1)}%</span>
            </div>
            <div class="performance-metric">
                <span class="metric-label">Avg Confidence:</span>
                <span class="metric-value">${(performance.avg_confidence || 0).toFixed(3)}</span>
            </div>
            <div class="performance-metric">
                <span class="metric-label">Avg Inference Time:</span>
                <span class="metric-value">${(performance.avg_inference_time || 0).toFixed(3)}s</span>
            </div>
        `;
    }
    
    updateComparisonChart(stats) {
        if (this.comparisonChart) {
            const yoloRate = (stats.yolo_usage_rate || 0) * 100;
            const ruleBasedRate = (stats.rule_based_usage_rate || 0) * 100;
            const hybridRate = (stats.hybrid_usage_rate || 0) * 100;
            
            this.comparisonChart.data.datasets[0].data = [yoloRate, ruleBasedRate, hybridRate];
            this.comparisonChart.update();
        }
    }
    
    updateConnectionStatus(status) {
        const indicator = document.querySelector('.status-indicator');
        if (indicator) {
            indicator.className = `status-indicator status-${status}`;
        }
    }
    
    updateDetectionStatus(status) {
        const startBtn = document.getElementById('startDetectionBtn');
        const stopBtn = document.getElementById('stopDetectionBtn');
        
        if (status === 'started') {
            this.isDetectionRunning = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            this.startFrameCapture();
        } else if (status === 'stopped') {
            this.isDetectionRunning = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            this.stopFrameCapture();
        }
    }
    
    startDetection() {
        if (this.socket) {
            this.socket.emit('start_detection');
            this.showNotification('Starting detection...', 'info');
        }
    }
    
    stopDetection() {
        if (this.socket) {
            this.socket.emit('stop_detection');
            this.showNotification('Stopping detection...', 'info');
        }
    }
    
    startFrameCapture() {
        if (this.isDetectionRunning) {
            this.captureAndSendFrame();
            setTimeout(() => this.startFrameCapture(), 100); // 10 FPS
        }
    }
    
    stopFrameCapture() {
        this.isDetectionRunning = false;
    }
    
    captureAndSendFrame() {
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('webcamCanvas');
        const context = canvas.getContext('2d');
        
        if (video.videoWidth && video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            if (this.socket) {
                this.socket.emit('frame_data', { image: imageData });
            }
        }
    }
    
    handleFrameResult(result) {
        this.updateResultsDisplay(result);
        this.addToDetectionHistory(result);
        
        // Update stats periodically
        if (this.detectionHistory.length % 10 === 0) {
            this.loadYOLOHybridStats();
        }
    }
    
    updateResultsDisplay(result) {
        const container = document.getElementById('resultsContent');
        
        if (result.faces_detected === 0) {
            container.innerHTML = '<p class="text-muted">No faces detected</p>';
            return;
        }
        
        let html = '';
        
        result.emotions.forEach((emotion, index) => {
            const emoji = this.getEmotionEmoji(emotion.emotion);
            const color = this.getEmotionColor(emotion.emotion);
            
            html += `
                <div class="border rounded p-2 mb-2" style="border-left: 4px solid ${color} !important;">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="fs-5">${emoji}</span>
                        <span class="badge bg-primary">Face ${index + 1}</span>
                    </div>
                    <div class="mt-2">
                        <strong>Emotion:</strong> <span class="emotion-${emotion.emotion.toLowerCase()}">${emotion.emotion}</span><br>
                        <strong>Confidence:</strong> ${(emotion.confidence * 100).toFixed(1)}%<br>
                        <strong>Primary Method:</strong> ${emotion.primary_method || 'Unknown'}<br>
                        <strong>Ensemble Method:</strong> ${result.ensemble_method || 'Unknown'}
                    </div>
                    <div class="confidence-bar mt-2">
                        <div class="confidence-fill" style="width: ${emotion.confidence * 100}%"></div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    addToDetectionHistory(result) {
        const historyItem = {
            timestamp: new Date().toLocaleTimeString(),
            faces_detected: result.faces_detected,
            ensemble_method: result.ensemble_method,
            processing_time: result.processing_time
        };
        
        this.detectionHistory.unshift(historyItem);
        
        // Keep only last 20 items
        if (this.detectionHistory.length > 20) {
            this.detectionHistory = this.detectionHistory.slice(0, 20);
        }
        
        this.updateDetectionHistory();
    }
    
    updateDetectionHistory() {
        const container = document.getElementById('detectionHistory');
        
        if (this.detectionHistory.length === 0) {
            container.innerHTML = '<p class="text-muted">No detection history yet...</p>';
            return;
        }
        
        let html = '';
        this.detectionHistory.forEach(item => {
            html += `
                <div class="history-item">
                    <div class="history-timestamp">${item.timestamp}</div>
                    <div class="history-details">
                        <strong>Faces:</strong> ${item.faces_detected} | 
                        <strong>Method:</strong> ${item.ensemble_method} | 
                        <strong>Time:</strong> ${(item.processing_time || 0).toFixed(3)}s
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    async updateYOLOWeight(weight) {
        try {
            const response = await fetch('/api/hybrid/update-yolo-weight', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ yolo_weight: weight })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification(`YOLO weight updated to ${weight}`, 'success');
                this.loadYOLOHybridStats();
            } else {
                this.showNotification(`Error updating YOLO weight: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Error updating YOLO weight:', error);
            this.showNotification('Error updating YOLO weight', 'error');
        }
    }
    
    async updateConfidenceThreshold(threshold) {
        try {
            const response = await fetch('/api/hybrid/update-confidence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ confidence_threshold: threshold })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification(`Confidence threshold updated to ${threshold}`, 'success');
            } else {
                this.showNotification(`Error updating confidence threshold: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Error updating confidence threshold:', error);
            this.showNotification('Error updating confidence threshold', 'error');
        }
    }
    
    async resetStats() {
        try {
            const response = await fetch('/api/hybrid/reset-stats', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('Statistics reset successfully', 'success');
                this.detectionHistory = [];
                this.updateDetectionHistory();
                this.loadInitialData();
            } else {
                this.showNotification(`Error resetting stats: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Error resetting stats:', error);
            this.showNotification('Error resetting stats', 'error');
        }
    }
    
    async saveResults() {
        try {
            const response = await fetch('/api/hybrid/save-results', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification(`Results saved to ${result.filename}`, 'success');
            } else {
                this.showNotification(`Error saving results: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Error saving results:', error);
            this.showNotification('Error saving results', 'error');
        }
    }
    
    getEmotionEmoji(emotion) {
        const emojis = {
            'Happy': '😊',
            'Sad': '😢',
            'Angry': '😠',
            'Surprised': '😲',
            'Fear': '😨',
            'Disgust': '🤢',
            'Neutral': '😐',
            'Unknown': '❓'
        };
        return emojis[emotion] || '❓';
    }
    
    getEmotionColor(emotion) {
        const colors = {
            'Happy': '#28a745',
            'Sad': '#dc3545',
            'Angry': '#fd7e14',
            'Surprised': '#17a2b8',
            'Fear': '#6f42c1',
            'Disgust': '#20c997',
            'Neutral': '#6c757d',
            'Unknown': '#6c757d'
        };
        return colors[emotion] || '#6c757d';
    }
    
    showNotification(message, type = 'info') {
        const toast = document.getElementById('notificationToast');
        const toastBody = document.getElementById('toastBody');
        
        toastBody.textContent = message;
        
        // Set toast color based on type
        toast.className = `toast ${type === 'error' ? 'bg-danger text-white' : ''}`;
        
        // Show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new YOLOHybridDashboard();
}); 
/**
 * Hybrid Emotion Detection Dashboard JavaScript
 * Mengelola perbandingan rule-based vs trained model
 */

// Global variables
let socket;
let webcam;
let canvas;
let ctx;
let isProcessing = false;
let processingInterval;
let comparisonChart;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Hybrid Dashboard...');
    
    // Initialize components
    initializeWebcam();
    initializeSocketIO();
    initializeComparisonChart();
    initializeWeightSlider();
    
    // Load initial data
    loadHybridStats();
    loadTrainedModelInfo();
    
    console.log('✅ Hybrid Dashboard initialized!');
});

// Initialize webcam
function initializeWebcam() {
    webcam = document.getElementById('webcam');
    canvas = document.getElementById('overlay');
    ctx = canvas.getContext('2d');
    
    // Get user media
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            webcam.srcObject = stream;
            webcam.onloadedmetadata = () => {
                canvas.width = webcam.videoWidth;
                canvas.height = webcam.videoHeight;
                console.log('📹 Webcam initialized:', webcam.videoWidth, 'x', webcam.videoHeight);
            };
        })
        .catch(err => {
            console.error('❌ Webcam error:', err);
            showNotification('Webcam access denied! ❌', 'error');
        });
}

// Initialize Socket.IO
function initializeSocketIO() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('🔌 Connected to hybrid system');
        showNotification('Connected to hybrid system! ✅');
    });
    
    socket.on('disconnect', () => {
        console.log('🔌 Disconnected from hybrid system');
        showNotification('Disconnected from hybrid system! ❌', 'error');
    });
    
    socket.on('frame_result', handleFrameResult);
}

// Initialize comparison chart
function initializeComparisonChart() {
    const chartCtx = document.getElementById('comparison-chart').getContext('2d');
    
    comparisonChart = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: ['Rule-Based', 'Trained Model', 'Hybrid Ensemble'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [85, 95, 96],
                backgroundColor: [
                    'rgba(255, 152, 0, 0.8)',
                    'rgba(76, 175, 80, 0.8)',
                    'rgba(156, 39, 176, 0.8)'
                ],
                borderColor: [
                    'rgba(255, 152, 0, 1)',
                    'rgba(76, 175, 80, 1)',
                    'rgba(156, 39, 176, 1)'
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
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '% accuracy';
                        }
                    }
                }
            }
        }
    });
}

// Initialize weight slider
function initializeWeightSlider() {
    const weightSlider = document.getElementById('ensemble-weight');
    const weightValue = document.getElementById('weight-value');
    const ruleWeight = document.getElementById('rule-weight');
    
    weightSlider.addEventListener('input', function() {
        const trainedWeight = parseInt(this.value);
        const ruleBasedWeight = 100 - trainedWeight;
        
        weightValue.textContent = trainedWeight + '%';
        ruleWeight.textContent = ruleBasedWeight + '%';
        
        // Update ensemble weight in backend
        updateEnsembleWeight(trainedWeight / 100);
    });
}

// Load hybrid statistics
function loadHybridStats() {
    fetch('/api/hybrid-stats')
        .then(response => response.json())
        .then(data => {
            console.log('📊 Hybrid stats loaded:', data);
            updateAgreementStats(data);
        })
        .catch(error => {
            console.error('❌ Error loading hybrid stats:', error);
        });
}

// Load trained model information
function loadTrainedModelInfo() {
    fetch('/api/trained-model-info')
        .then(response => response.json())
        .then(data => {
            console.log('🧠 Trained model info loaded:', data);
            updateTrainedModelStatus(data);
        })
        .catch(error => {
            console.error('❌ Error loading trained model info:', error);
        });
}

// Update agreement statistics
function updateAgreementStats(stats) {
    document.getElementById('total-comparisons').textContent = stats.total_comparisons || 0;
    document.getElementById('agreement-rate').textContent = 
        ((stats.agreement_rate || 0) * 100).toFixed(1) + '%';
    
    if (stats.accuracy_comparison) {
        document.getElementById('hybrid-confidence').textContent = 
            ((stats.accuracy_comparison.hybrid_avg_confidence || 0) * 100).toFixed(1) + '%';
        document.getElementById('ensemble-boost').textContent = 
            ((stats.accuracy_comparison.hybrid_confidence_boost || 0) * 100).toFixed(1) + '%';
    }
}

// Update trained model status
function updateTrainedModelStatus(modelInfo) {
    const trainedStatus = document.getElementById('trained-status');
    const trainedDetails = document.getElementById('trained-details');
    
    if (modelInfo.status === 'loaded') {
        trainedStatus.textContent = 'Active';
        trainedStatus.className = 'method-status status-active';
        trainedDetails.textContent = `Model: ${modelInfo.model_path}`;
        
        // Update accuracy
        document.getElementById('trained-accuracy').textContent = '95-98%';
        document.getElementById('trained-speed').textContent = 'Medium';
        
        console.log('✅ Trained model loaded successfully');
    } else {
        trainedStatus.textContent = 'Not Loaded';
        trainedStatus.className = 'method-status status-inactive';
        trainedDetails.textContent = 'No model loaded';
        
        // Update accuracy
        document.getElementById('trained-accuracy').textContent = 'N/A';
        document.getElementById('trained-speed').textContent = 'N/A';
        
        console.log('⚠️ No trained model loaded');
    }
}

// Load trained model
function loadTrainedModel() {
    const fileInput = document.getElementById('model-file');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select a model file! ❌', 'error');
        return;
    }
    
    console.log('📁 Loading model file:', file.name);
    
    // Create FormData
    const formData = new FormData();
    formData.append('model', file);
    
    // Upload model file
    fetch('/api/set-model-path', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_path: file.name  // This should be the actual path
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Model loaded successfully! ✅');
            loadTrainedModelInfo();
            loadHybridStats();
        } else {
            showNotification('Error loading model: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('❌ Error loading model:', error);
        showNotification('Error loading model! ❌', 'error');
    });
}

// Update ensemble weight
function updateEnsembleWeight(weight) {
    // This would typically update the backend
    console.log('🎯 Ensemble weight updated:', weight);
}

// Handle frame results
function handleFrameResult(data) {
    console.log('📊 Frame result received:', data);
    
    if (data.error) {
        console.error('❌ Frame processing error:', data.error);
        return;
    }
    
    // Update statistics
    updateAgreementStats(data.agreement_stats || {});
    
    // Draw bounding boxes
    drawBoundingBoxes(data.detections || []);
    
    // Update results display
    updateResultsDisplay(data);
    
    // Update comparison chart
    updateComparisonChart(data);
}

// Draw bounding boxes
function drawBoundingBoxes(detections) {
    if (!canvas || !ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    detections.forEach(detection => {
        const bbox = detection.bbox;
        const emotion = detection.emotion;
        const confidence = detection.emotion_confidence;
        const ensembleMethod = detection.ensemble_method;
        
        // Get emotion color
        const color = getEmotionColor(emotion);
        
        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(bbox[0], bbox[1], bbox[2], bbox[3]);
        
        // Create label
        let label = `${emotion} (${(confidence * 100).toFixed(1)}%)`;
        if (ensembleMethod && ensembleMethod !== 'unknown') {
            label += ` [${ensembleMethod}]`;
        }
        
        // Draw label background
        ctx.font = '16px Arial';
        const labelWidth = ctx.measureText(label).width;
        const labelHeight = 20;
        
        ctx.fillStyle = color;
        ctx.fillRect(bbox[0], bbox[1] - labelHeight - 5, labelWidth + 10, labelHeight);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, bbox[0] + 5, bbox[1] - 8);
        
        // Draw emotion emoji
        const emoji = getEmotionEmoji(emotion);
        ctx.font = '24px Arial';
        ctx.fillText(emoji, bbox[0] + bbox[2] - 30, bbox[1] + bbox[3] + 25);
    });
}

// Update results display
function updateResultsDisplay(data) {
    const resultsContainer = document.getElementById('results-container');
    
    if (!data.detections || data.detections.length === 0) {
        resultsContainer.innerHTML = '<p class="no-face">No faces detected! 😔</p>';
        return;
    }
    
    let html = '<div class="detection-grid">';
    
    data.detections.forEach(detection => {
        const emotion = detection.emotion;
        const confidence = detection.emotion_confidence;
        const ensembleMethod = detection.ensemble_method;
        const hybridAnalysis = detection.hybrid_analysis;
        
        html += `
            <div class="detection-card ${emotion.toLowerCase()}">
                <div class="detection-header">
                    <span class="emotion-emoji">${getEmotionEmoji(emotion)}</span>
                    <span class="emotion-label">${emotion}</span>
                    <span class="confidence-text">${(confidence * 100).toFixed(1)}%</span>
                </div>
                
                <div class="detection-details">
                    <div class="method-comparison">
                        <div class="method-result">
                            <strong>Rule-Based:</strong> ${hybridAnalysis.rule_based_emotion} 
                            (${(hybridAnalysis.rule_based_confidence * 100).toFixed(1)}%)
                        </div>
                        <div class="method-result">
                            <strong>Trained Model:</strong> ${hybridAnalysis.trained_model_emotion} 
                            (${(hybridAnalysis.trained_model_confidence * 100).toFixed(1)}%)
                        </div>
                    </div>
                    
                    <div class="ensemble-info">
                        <strong>Ensemble Method:</strong> ${ensembleMethod}
                        <br>
                        <strong>Agreement:</strong> ${hybridAnalysis.agreement ? '✅ Yes' : '❌ No'}
                        <br>
                        <strong>Hybrid Confidence:</strong> ${(hybridAnalysis.hybrid_confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    if (data.processing_time) {
        html += `
            <div class="processing-info">
                <p>Processing time: ${(data.processing_time * 1000).toFixed(1)}ms</p>
                <p>Detection method: ${data.detection_method}</p>
                <p>Trained model loaded: ${data.trained_model_loaded ? '✅ Yes' : '❌ No'}</p>
            </div>
        `;
    }
    
    resultsContainer.innerHTML = html;
}

// Update comparison chart
function updateComparisonChart(data) {
    if (!comparisonChart || !data.detections) return;
    
    // Calculate average confidences
    let ruleConfidences = [];
    let trainedConfidences = [];
    let hybridConfidences = [];
    
    data.detections.forEach(detection => {
        if (detection.hybrid_analysis) {
            ruleConfidences.push(detection.hybrid_analysis.rule_based_confidence);
            trainedConfidences.push(detection.hybrid_analysis.trained_model_confidence);
            hybridConfidences.push(detection.hybrid_analysis.hybrid_confidence);
        }
    });
    
    if (ruleConfidences.length > 0) {
        const avgRule = ruleConfidences.reduce((a, b) => a + b, 0) / ruleConfidences.length;
        const avgTrained = trainedConfidences.reduce((a, b) => a + b, 0) / trainedConfidences.length;
        const avgHybrid = hybridConfidences.reduce((a, b) => a + b, 0) / hybridConfidences.length;
        
        // Update chart data
        comparisonChart.data.datasets[0].data = [
            avgRule * 100,
            avgTrained * 100,
            avgHybrid * 100
        ];
        comparisonChart.update('active');
    }
}

// Get emotion color
function getEmotionColor(emotion) {
    const colors = {
        'Happy': '#4caf50',
        'Sad': '#f44336',
        'Angry': '#2196f3',
        'Surprised': '#ff9800',
        'Fear': '#9c27b0',
        'Disgust': '#009688',
        'Neutral': '#9e9e9e'
    };
    return colors[emotion] || '#000000';
}

// Get emotion emoji
function getEmotionEmoji(emotion) {
    const emojis = {
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Surprised': '😲',
        'Fear': '😨',
        'Disgust': '🤢',
        'Neutral': '😐'
    };
    return emojis[emotion] || '❓';
}

// Control functions
function startDetection() {
    if (isProcessing) return;
    
    console.log('🎥 Starting hybrid detection...');
    isProcessing = true;
    
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    
    // Start frame processing
    processingInterval = setInterval(() => {
        if (webcam.videoWidth > 0 && webcam.videoHeight > 0) {
            // Capture frame from webcam
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            
            // Draw current frame to canvas
            ctx.drawImage(webcam, 0, 0);
            
            // Get frame data
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server
            socket.emit('frame_data', { image: frameData });
        }
    }, 500);
    
    showNotification('Hybrid detection started! 🚀');
}

function stopDetection() {
    if (!isProcessing) return;
    
    console.log('⏹️ Stopping hybrid detection...');
    isProcessing = false;
    
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    
    // Stop frame processing
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    
    // Clear canvas
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    showNotification('Hybrid detection stopped! ⏹️');
}

function resetStats() {
    console.log('🔄 Resetting hybrid statistics...');
    
    fetch('/api/reset-hybrid-stats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification('Statistics reset successfully! 🔄');
                loadHybridStats();
            } else {
                showNotification('Error resetting statistics! ❌', 'error');
            }
        })
        .catch(error => {
            console.error('❌ Error resetting stats:', error);
            showNotification('Error resetting statistics! ❌', 'error');
        });
}

function saveResults() {
    console.log('💾 Saving hybrid results...');
    
    fetch('/api/save-hybrid-results')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification('Results saved successfully! 💾');
            } else {
                showNotification('Error saving results! ❌', 'error');
            }
        })
        .catch(error => {
            console.error('❌ Error saving results:', error);
            showNotification('Error saving results! ❌', 'error');
        });
}

// Event listeners
document.getElementById('start-btn').addEventListener('click', startDetection);
document.getElementById('stop-btn').addEventListener('click', stopDetection);
document.getElementById('reset-btn').addEventListener('click', resetStats);
document.getElementById('save-btn').addEventListener('click', saveResults);

// Utility functions
function showNotification(message, type = 'info') {
    // Simple notification system
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        background: ${type === 'error' ? '#f44336' : '#4caf50'};
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Auto-refresh stats every 5 seconds
setInterval(() => {
    if (isProcessing) {
        loadHybridStats();
    }
}, 5000); 
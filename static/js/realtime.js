// Advanced Real-time Emotion Detection with Charts and Visualizations
let stream = null;
let canvas = null;
let ctx = null;
let isProcessing = false;
let emotionHistory = {};
let chart = null;
let personCounter = 0;

// Chart.js untuk realtime emotion distribution
let emotionChart = null;
let emotionData = {
    labels: ['Happy', 'Neutral', 'Sad', 'Angry', 'Surprised'],
    datasets: [{
        label: 'Current Emotions',
        data: [0, 0, 0, 0, 0],
        backgroundColor: [
            'rgba(0, 255, 0, 0.8)',   // Happy - Green
            'rgba(0, 0, 255, 0.8)',   // Neutral - Blue
            'rgba(128, 128, 128, 0.8)', // Sad - Gray
            'rgba(255, 0, 0, 0.8)',   // Angry - Red
            'rgba(255, 0, 255, 0.8)'  // Surprised - Magenta
        ],
        borderColor: [
            'rgba(0, 255, 0, 1)',
            'rgba(0, 0, 255, 1)',
            'rgba(128, 128, 128, 1)',
            'rgba(255, 0, 0, 1)',
            'rgba(255, 0, 255, 1)'
        ],
        borderWidth: 2
    }]
};

// Initialize realtime dashboard
function initializeRealtimeDashboard() {
    // Create canvas overlay untuk bounding boxes
    const videoContainer = document.querySelector('.video-container');
    canvas = document.createElement('canvas');
    canvas.id = 'overlay-canvas';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '10';
    
    videoContainer.appendChild(canvas);
    ctx = canvas.getContext('2d');
    
    // Initialize Chart.js
    initializeEmotionChart();
    
    // Initialize statistics dashboard
    initializeStatsDashboard();
    
    console.log('✅ Realtime dashboard initialized');
}

// Initialize emotion distribution chart
function initializeEmotionChart() {
    const chartCanvas = document.getElementById('emotion-chart');
    if (!chartCanvas) return;
    
    emotionChart = new Chart(chartCanvas, {
        type: 'doughnut',
        data: emotionData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#ffffff',
                        font: {
                            size: 12
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Real-time Emotion Distribution',
                    color: '#ffffff',
                    font: {
                        size: 16
                    }
                }
            },
            animation: {
                duration: 500,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Initialize statistics dashboard
function initializeStatsDashboard() {
    updateStatsDisplay({
        total_faces: 0,
        processing_time: 0,
        dominant_emotion: 'None',
        emotion_stability: 0
    });
}

// Update statistics display
function updateStatsDisplay(stats) {
    const statsContainer = document.getElementById('stats-container');
    if (!statsContainer) return;
    
    statsContainer.innerHTML = `
        <div class="stat-card">
            <div class="stat-icon">👥</div>
            <div class="stat-value">${stats.total_faces}</div>
            <div class="stat-label">Faces Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">${(stats.processing_time * 1000).toFixed(1)}ms</div>
            <div class="stat-label">Processing Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">${stats.dominant_emotion}</div>
            <div class="stat-label">Dominant Emotion</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📊</div>
            <div class="stat-value">${(stats.emotion_stability * 100).toFixed(1)}%</div>
            <div class="stat-label">Emotion Stability</div>
        </div>
    `;
}

// Update emotion chart data
function updateEmotionChart(detections) {
    if (!emotionChart) return;
    
    // Reset emotion counts
    const emotionCounts = [0, 0, 0, 0, 0];
    
    // Count emotions from detections
    detections.forEach(detection => {
        const emotion = detection.emotion;
        const emotionIndex = emotionData.labels.indexOf(emotion);
        if (emotionIndex !== -1) {
            emotionCounts[emotionIndex]++;
        }
    });
    
    // Update chart data
    emotionChart.data.datasets[0].data = emotionCounts;
    emotionChart.update('none'); // Update without animation for real-time
}

// Draw bounding boxes dengan labels yang jelas
function drawBoundingBoxes(detections, videoElement) {
    if (!canvas || !ctx || !videoElement) return;
    
    // Set canvas size to match video
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    detections.forEach((detection, index) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const emotion = detection.emotion;
        const confidence = detection.emotion_confidence;
        const trackId = detection.track_id;
        
        // Calculate colors based on emotion
        const colors = getEmotionColors(emotion);
        
        // Draw bounding box
        ctx.strokeStyle = colors.border;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw filled background untuk label
        const labelText = `${emotion} (${(confidence * 100).toFixed(1)}%)`;
        const labelWidth = ctx.measureText(labelText).width + 20;
        const labelHeight = 30;
        
        ctx.fillStyle = colors.background;
        ctx.fillRect(x1, y1 - labelHeight, labelWidth, labelHeight);
        
        // Draw label text
        ctx.fillStyle = colors.text;
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(labelText, x1 + 10, y1 - 10);
        
        // Draw person ID
        ctx.fillStyle = colors.border;
        ctx.font = 'bold 12px Arial';
        ctx.fillText(`ID: ${trackId}`, x1, y2 + 20);
        
        // Draw emotion icon
        const emoji = getEmotionEmoji(emotion);
        ctx.font = '20px Arial';
        ctx.fillText(emoji, x1 + labelWidth + 10, y1 - 10);
    });
}

// Get emotion colors
function getEmotionColors(emotion) {
    const colorMap = {
        'Happy': {
            border: '#00FF00',
            background: 'rgba(0, 255, 0, 0.9)',
            text: '#000000'
        },
        'Neutral': {
            border: '#0080FF',
            background: 'rgba(0, 128, 255, 0.9)',
            text: '#FFFFFF'
        },
        'Sad': {
            border: '#808080',
            background: 'rgba(128, 128, 128, 0.9)',
            text: '#FFFFFF'
        },
        'Angry': {
            border: '#FF0000',
            background: 'rgba(255, 0, 0, 0.9)',
            text: '#FFFFFF'
        },
        'Surprised': {
            border: '#FF00FF',
            background: 'rgba(255, 0, 255, 0.9)',
            text: '#FFFFFF'
        }
    };
    
    return colorMap[emotion] || colorMap['Neutral'];
}

// Get emotion emoji
function getEmotionEmoji(emotion) {
    const emojiMap = {
        'Happy': '😊',
        'Neutral': '😐',
        'Sad': '😢',
        'Angry': '😠',
        'Surprised': '😲'
    };
    
    return emojiMap[emotion] || '❓';
}

// Update emotion timeline
function updateEmotionTimeline(detections) {
    const timelineContainer = document.getElementById('emotion-timeline');
    if (!timelineContainer) return;
    
    // Update emotion history
    detections.forEach(detection => {
        const trackId = detection.track_id;
        if (!emotionHistory[trackId]) {
            emotionHistory[trackId] = [];
        }
        
        emotionHistory[trackId].push({
            emotion: detection.emotion,
            timestamp: Date.now(),
            confidence: detection.emotion_confidence
        });
        
        // Keep only last 10 emotions per person
        if (emotionHistory[trackId].length > 10) {
            emotionHistory[trackId] = emotionHistory[trackId].slice(-10);
        }
    });
    
    // Render timeline
    renderEmotionTimeline();
}

// Render emotion timeline
function renderEmotionTimeline() {
    const timelineContainer = document.getElementById('emotion-timeline');
    if (!timelineContainer) return;
    
    let timelineHTML = '<h3>Emotion Timeline</h3>';
    
    Object.entries(emotionHistory).forEach(([trackId, emotions]) => {
        timelineHTML += `
            <div class="person-timeline">
                <div class="person-header">
                    <span class="person-id">${trackId}</span>
                    <span class="emotion-count">${emotions.length} emotions</span>
                </div>
                <div class="emotion-sequence">
                    ${emotions.map(emotion => `
                        <span class="emotion-badge ${emotion.emotion.toLowerCase()}" 
                              title="${emotion.emotion} (${(emotion.confidence * 100).toFixed(1)}%)">
                            ${getEmotionEmoji(emotion.emotion)}
                        </span>
                    `).join('')}
                </div>
            </div>
        `;
    });
    
    timelineContainer.innerHTML = timelineHTML;
}

// Toggle webcam
async function toggleWebcam() {
    const webcamToggle = document.getElementById('webcam-toggle');
    const webcamOff = document.getElementById('webcam-off');
    const loading = document.getElementById('loading');
    const video = document.getElementById('webcam');
    const resultDiv = document.getElementById('realtime-result');
    
    if (stream) {
        // Turn off webcam
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        webcamOff.style.display = 'block';
        video.style.display = 'none';
        webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn On Webcam';
        resultDiv.innerHTML = '';
        
        // Clear canvas
        if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        
        // Reset emotion history
        emotionHistory = {};
        updateEmotionChart([]);
        updateStatsDisplay({
            total_faces: 0,
            processing_time: 0,
            dominant_emotion: 'None',
            emotion_stability: 0
        });
        
    } else {
        // Turn on webcam
        webcamOff.style.display = 'none';
        loading.style.display = 'block';
        webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn Off Webcam';
        
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                loading.style.display = 'none';
                video.style.display = 'block';
                
                // Start real-time processing
                startRealtimeProcessing();
                
                showNotification('Webcam activated! 🎥');
            };
            
        } catch (error) {
            loading.style.display = 'none';
            webcamOff.style.display = 'block';
            webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn On Webcam';
            resultDiv.innerHTML = `<p class="error-message">Cannot access webcam: ${error.message} 🚫</p>`;
        }
    }
}

// Start real-time processing
function startRealtimeProcessing() {
    if (isProcessing) return;
    isProcessing = true;
    
    const video = document.getElementById('webcam');
    
    function processFrame() {
        if (!isProcessing || !stream) return;
        
        // Create canvas untuk capture frame
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        
        // Draw current video frame
        tempCtx.drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
        
        // Send frame untuk processing
        socket.emit('process_frame', { image: imageData });
        
        // Continue processing
        requestAnimationFrame(processFrame);
    }
    
    // Start processing loop
    processFrame();
}

// Handle frame results
function handleFrameResult(data) {
    if (data.error) {
        console.error('Frame processing error:', data.error);
        return;
    }

    const results = data.results;
    const detections = results.detections || [];
    
    // Update statistics
    updateStatsDisplay({
        total_faces: results.total_faces || 0,
        processing_time: results.processing_time || 0,
        dominant_emotion: getDominantEmotion(detections),
        emotion_stability: calculateEmotionStability(detections)
    });
    
    // Update emotion chart
    updateEmotionChart(detections);
    
    // Update emotion timeline
    updateEmotionTimeline(detections);
    
    // Draw bounding boxes
    const video = document.getElementById('webcam');
    drawBoundingBoxes(detections, video);
    
    // Update result display
    updateResultDisplay(results);
}

// Get dominant emotion
function getDominantEmotion(detections) {
    if (detections.length === 0) return 'None';
    
    const emotionCounts = {};
    detections.forEach(detection => {
        const emotion = detection.emotion;
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
    });
    
    return Object.entries(emotionCounts)
        .sort(([,a], [,b]) => b - a)[0][0];
}

// Calculate emotion stability
function calculateEmotionStability(detections) {
    if (detections.length === 0) return 0;
    
    let totalStability = 0;
    detections.forEach(detection => {
        if (detection.temporal_analysis && detection.temporal_analysis.emotion_stability) {
            totalStability += detection.temporal_analysis.emotion_stability;
        }
    });
    
    return totalStability / detections.length;
}

// Update result display
function updateResultDisplay(results) {
    const resultDiv = document.getElementById('realtime-result');
    if (!resultDiv) return;
    
    const detections = results.detections || [];
    
    if (detections.length === 0) {
        resultDiv.innerHTML = '<p class="no-faces">No faces detected</p>';
        return;
    }
    
    let resultHTML = '<div class="detection-results">';
    detections.forEach(detection => {
        const emotion = detection.emotion;
        const confidence = detection.emotion_confidence;
        const bbox = detection.bbox;
        
        resultHTML += `
            <div class="detection-item ${emotion.toLowerCase()}">
                <div class="detection-header">
                    <span class="emotion-emoji">${getEmotionEmoji(emotion)}</span>
                    <span class="emotion-label">${emotion}</span>
                    <span class="confidence">${(confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="detection-details">
                    <span class="bbox-info">BBox: [${bbox.join(', ')}]</span>
                    <span class="track-id">ID: ${detection.track_id}</span>
                </div>
            </div>
        `;
    });
    resultHTML += '</div>';
    
    resultDiv.innerHTML = resultHTML;
}

// Show notification
function showNotification(message) {
    const notification = document.getElementById('notification');
    if (notification) {
        notification.textContent = message;
        notification.style.display = 'block';
        setTimeout(() => {
            notification.style.display = 'none';
        }, 3000);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initializeRealtimeDashboard();
    
    // Setup event listeners
    const webcamToggle = document.getElementById('webcam-toggle');
    if (webcamToggle) {
        webcamToggle.addEventListener('click', toggleWebcam);
    }
    
    // Setup Socket.IO
    if (typeof io !== 'undefined') {
        socket = io();
        
        socket.on('connect', () => {
            console.log('Connected to server');
            showNotification('Connected to server! 🚀');
        });
        
        socket.on('frame_result', handleFrameResult);
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            showNotification('Disconnected from server! 📡');
        });
    }
    
    console.log('✅ Realtime page initialized');
});
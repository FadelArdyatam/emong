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
let temporalChart = null; // Chart untuk temporal data
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

// Temporal chart data
let temporalData = {
    labels: [],
    datasets: []
};

// Initialize realtime dashboard
function initializeRealtimeDashboard() {
    console.log('🔧 Starting dashboard initialization...');
    
    // Create canvas overlay untuk bounding boxes
    const videoContainer = document.querySelector('.video-container');
    console.log('📹 Video container found:', videoContainer);
    
    if (!videoContainer) {
        console.error('❌ Video container not found!');
        return;
    }
    
    canvas = document.createElement('canvas');
    canvas.id = 'overlay-canvas';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '10';
    
    videoContainer.appendChild(canvas);
    ctx = canvas.getContext('2d');
    
    console.log('🎨 Canvas overlay created and attached');
    
    // Initialize Chart.js
    initializeEmotionChart();
    initializeTemporalChart(); // Initialize temporal chart
    
    // Initialize statistics dashboard
    initializeStatsDashboard();
    
    // Initialize storage info
    updateStorageInfo();
    
    // Initialize emotion stats
    updateEmotionStats();
    
    // Add event listeners for emotion bias controls
    document.getElementById('reset-bias-button')?.addEventListener('click', resetEmotionBias);
    document.getElementById('balance-weights-button')?.addEventListener('click', balanceEmotionWeights);
    
    console.log('✅ Realtime dashboard initialized successfully!');
}

// Initialize emotion distribution chart
function initializeEmotionChart() {
    const chartCanvas = document.getElementById('emotion-chart');
    if (!chartCanvas) {
        console.error('❌ Chart canvas not found!');
        return;
    }
    
    console.log('📊 Initializing emotion chart...');
    
    // Destroy existing chart if any
    if (emotionChart) {
        emotionChart.destroy();
    }
    
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
                duration: 300,
                easing: 'easeInOutQuart'
            },
            elements: {
                arc: {
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }
            }
        }
    });
    
    console.log('✅ Emotion chart initialized successfully');
}

// Initialize temporal chart
function initializeTemporalChart() {
    const chartCanvas = document.getElementById('temporal-chart');
    if (!chartCanvas) {
        console.error('❌ Temporal chart canvas not found!');
        return;
    }

    console.log('📊 Initializing temporal chart...');

    // Destroy existing chart if any
    if (temporalChart) {
        temporalChart.destroy();
    }

    // Initialize temporal data
    temporalData = {
        labels: [],
        datasets: [{
            label: 'Emotion Stability',
            data: [],
            borderColor: '#ffd700',
            backgroundColor: 'rgba(255, 215, 0, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4
        }]
    };

    temporalChart = new Chart(chartCanvas, {
        type: 'line',
        data: temporalData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#ffffff',
                        font: {
                            size: 12
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Emotion Stability Over Time',
                    color: '#ffffff',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#ffffff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.2)'
                    }
                },
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        color: '#ffffff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.2)'
                    }
                }
            },
            animation: {
                duration: 300,
                easing: 'easeInOutQuart'
            },
            elements: {
                line: {
                    borderWidth: 2,
                    borderColor: '#ffd700'
                }
            }
        }
    });

    console.log('✅ Temporal chart initialized successfully');
}

// Update storage info
function updateStorageInfo() {
    const storageContainer = document.getElementById('storage-info');
    if (!storageContainer) return;

    // Fetch storage info from backend
    fetch('/api/storage-info')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                storageContainer.innerHTML = `<p style="color: #ff6b6b;">Error: ${data.error}</p>`;
                return;
            }

            const html = `
                <div class="storage-item">
                    <span class="storage-label">Data File:</span>
                    <span class="storage-value">${data.data_file}</span>
                </div>
                <div class="storage-item">
                    <span class="storage-label">File Exists:</span>
                    <span class="storage-value">${data.file_exists ? 'Yes' : 'No'}</span>
                </div>
                <div class="storage-item">
                    <span class="storage-label">File Size:</span>
                    <span class="storage-value">${data.file_size_mb} MB</span>
                </div>
                <div class="storage-item">
                    <span class="storage-label">Faces Tracked:</span>
                    <span class="storage-value">${data.total_faces_tracked}</span>
                </div>
                <div class="storage-item">
                    <span class="storage-label">Total Records:</span>
                    <span class="storage-value">${data.total_emotion_records}</span>
                </div>
                <div class="storage-item">
                    <span class="storage-label">Last Updated:</span>
                    <span class="storage-value">${new Date(data.last_updated).toLocaleString()}</span>
                </div>
            `;
            
            storageContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error fetching storage info:', error);
            storageContainer.innerHTML = '<p style="color: #ff6b6b;">Error loading storage info</p>';
        });
}

// Update emotion distribution stats
function updateEmotionStats() {
    const statsContainer = document.getElementById('emotion-stats-info');
    if (!statsContainer) return;

    // Fetch emotion distribution stats from backend
    fetch('/api/emotion-distribution')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                statsContainer.innerHTML = `<p style="color: #ff6b6b;">Error: ${data.error}</p>`;
                return;
            }

            if (data.message) {
                statsContainer.innerHTML = `<p style="color: #ccc;">${data.message}</p>`;
                return;
            }

            const emojiMap = {
                'Happy': '😊',
                'Neutral': '😐',
                'Sad': '😢',
                'Angry': '😠',
                'Surprised': '😲'
            };

            let html = `
                <div class="emotion-item">
                    <span class="emotion-label">Total Detections:</span>
                    <span class="emotion-value">${data.total_detections}</span>
                </div>
                <div class="emotion-item">
                    <span class="emotion-label">Most Common:</span>
                    <span class="emotion-value">${emojiMap[data.most_common_emotion]} ${data.most_common_emotion}</span>
                </div>
                <div class="emotion-item">
                    <span class="emotion-label">Balanced:</span>
                    <span class="emotion-value">${data.distribution_balanced ? '✅ Yes' : '⚠️ No'}</span>
                </div>
            `;

            // Add emotion percentages
            Object.entries(data.emotion_percentages).forEach(([emotion, percentage]) => {
                const emoji = emojiMap[emotion] || '❓';
                const biasInfo = data.bias_analysis[emotion];
                const isHighBias = percentage > 30;
                
                html += `
                    <div class="emotion-item ${emotion.toLowerCase()} ${isHighBias ? 'bias-warning' : ''}">
                        <span class="emotion-label">
                            ${emoji} ${emotion}
                        </span>
                        <span class="emotion-value">
                            ${percentage.toFixed(1)}% (${data.emotion_counts[emotion]})
                        </span>
                    </div>
                `;
            });

            // Add bias warnings
            const highBiasEmotions = Object.entries(data.bias_analysis)
                .filter(([emotion, info]) => info.includes('High bias'))
                .map(([emotion]) => emotion);

            if (highBiasEmotions.length > 0) {
                html += `
                    <div class="emotion-item bias-warning">
                        <span class="emotion-label">⚠️ High Bias Detected:</span>
                        <span class="emotion-value">${highBiasEmotions.join(', ')}</span>
                    </div>
                `;
            }
            
            statsContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error fetching emotion stats:', error);
            statsContainer.innerHTML = '<p style="color: #ff6b6b;">Error loading emotion stats</p>';
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
    
    const detectionMethod = stats.mediapipe_enabled ? 
        '<span style="color: #4caf50;">✅ Advanced</span>' : 
        '<span style="color: #ff9800;">⚠️ Traditional</span>';
    
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
            <div class="stat-icon">🔬</div>
            <div class="stat-value">${detectionMethod}</div>
            <div class="stat-label">Detection Method</div>
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
    if (!emotionChart) {
        console.warn('⚠️ Chart not initialized');
        return;
    }
    
    console.log('📊 Updating emotion chart with detections:', detections);
    
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
    
    console.log('📈 New emotion counts:', emotionCounts);
    
    // Update chart data with smooth animation
    emotionChart.data.datasets[0].data = emotionCounts;
    
    // Update chart with animation
    emotionChart.update('active');
    
    console.log('✅ Chart updated successfully');
}

// Update temporal chart data
function updateTemporalChart(detections) {
    if (!temporalChart) {
        console.warn('⚠️ Temporal chart not initialized');
        return;
    }

    const currentTime = Date.now();
    const emotionStability = calculateEmotionStability(detections);

    temporalData.labels.push(new Date(currentTime).toLocaleTimeString());
    temporalData.datasets[0].data.push(emotionStability);

    // Keep only last 100 data points for temporal chart
    if (temporalData.labels.length > 100) {
        temporalData.labels.shift();
        temporalData.datasets[0].data.shift();
    }

    temporalChart.update();
}

// Draw bounding boxes dengan labels yang jelas
function drawBoundingBoxes(detections) {
    if (!canvas || !ctx) {
        console.warn('⚠️ Canvas not initialized');
        return;
    }
    
    const videoElement = document.getElementById('webcam');
    if (!videoElement) {
        console.warn('⚠️ Video element not found');
        return;
    }
    
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
        
        // Calculate scaling factors if image dimensions are provided
        let scaleX = 1, scaleY = 1;
        if (detection.image_width && detection.image_height) {
            scaleX = canvas.width / detection.image_width;
            scaleY = canvas.height / detection.image_height;
        }
        
        // Apply scaling to bounding box coordinates
        const scaledX1 = x1 * scaleX;
        const scaledY1 = y1 * scaleY;
        const scaledX2 = x2 * scaleX;
        const scaledY2 = y2 * scaleY;
        
        // Calculate colors based on emotion
        const colors = getEmotionColors(emotion);
        
        // Draw bounding box
        ctx.strokeStyle = colors.border;
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
        
        // Draw filled background untuk label
        const labelText = `${emotion} (${(confidence * 100).toFixed(1)}%)`;
        const labelWidth = ctx.measureText(labelText).width + 20;
        const labelHeight = 30;
        
        ctx.fillStyle = colors.background;
        ctx.fillRect(scaledX1, scaledY1 - labelHeight, labelWidth, labelHeight);
        
        // Draw label text
        ctx.fillStyle = colors.text;
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(labelText, scaledX1 + 10, scaledY1 - 10);
        
        // Draw person ID
        ctx.fillStyle = colors.border;
        ctx.font = 'bold 12px Arial';
        ctx.fillText(`ID: ${trackId}`, scaledX1, scaledY2 + 20);
        
        // Draw emotion icon
        const emoji = getEmotionEmoji(emotion);
        ctx.font = '20px Arial';
        ctx.fillText(emoji, scaledX1 + labelWidth + 10, scaledY1 - 10);
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
        updateTemporalChart([]); // Clear temporal chart
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
    console.log('🚀 Starting real-time processing...');
    
    if (isProcessing) {
        console.log('⚠️ Already processing, skipping...');
        return;
    }
    
    isProcessing = true;
    console.log('✅ Processing flag set to true');
    
    const video = document.getElementById('webcam');
    console.log('📹 Video element:', video);
    console.log('📐 Video dimensions:', video.videoWidth, 'x', video.videoHeight);
    
    // Set processing interval (every 500ms = 2 FPS)
    const processingInterval = setInterval(() => {
        if (!isProcessing || !stream) {
            console.log('⏹️ Processing stopped or stream ended');
            clearInterval(processingInterval);
            return;
        }
        
        // Create canvas untuk capture frame
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        
        // Draw current video frame
        tempCtx.drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
        
        console.log('📤 Sending frame for processing, size:', imageData.length);
        
        // Send frame untuk processing
        if (socket && socket.connected) {
            socket.emit('process_frame', { image: imageData });
        } else {
            console.warn('⚠️ Socket not connected, skipping frame');
        }
        
    }, 500); // Process every 500ms
    
    console.log('🔄 Started processing loop with 500ms interval');
}

// Handle frame results
function handleFrameResult(data) {
    console.log('📡 Frame result received:', data);
    
    if (data.error) {
        console.error('❌ Frame processing error:', data.error);
        return;
    }
    
    const detections = data.detections || [];
    const totalFaces = data.total_faces || 0;
    const processingTime = data.processing_time || 0;
    
    console.log('👥 Detections found:', detections.length);
    console.log('📊 Results:', data);
    
    // Update statistics with accuracy info
    updateStatsDisplay({
        total_faces: totalFaces,
        processing_time: processingTime,
        mediapipe_enabled: data.mediapipe_enabled,
        dominant_emotion: getDominantEmotion(detections),
        emotion_stability: calculateEmotionStability(detections)
    });
    
    // Update emotion chart
    updateEmotionChart(detections);
    
    // Update temporal chart
    updateTemporalChart(detections);
    
    // Update emotion timeline
    updateEmotionTimeline(detections);
    
    // Draw bounding boxes
    const video = document.getElementById('webcam');
    console.log('🎥 Drawing bounding boxes for video:', video);
    drawBoundingBoxes(detections);
    
    // Update result display
    updateResultDisplay(data);
    
    // Show notification if faces detected
    if (totalFaces > 0) {
        showNotification(`Detected ${totalFaces} face(s)! 👥`);
    }
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

// Calculate emotion stability for temporal chart
function calculateEmotionStability(detections) {
    if (!detections || detections.length === 0) {
        return 0;
    }
    
    // Calculate average confidence as stability indicator
    const confidences = detections.map(d => d.emotion_confidence);
    const avgConfidence = confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
    
    // Calculate emotion consistency
    const emotions = detections.map(d => d.emotion);
    const uniqueEmotions = new Set(emotions);
    const consistency = 1 - (uniqueEmotions.size - 1) / (emotions.length - 1);
    
    // Combine confidence and consistency
    return (avgConfidence + consistency) / 2;
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
    
    // Setup confidence slider
    const confidenceSlider = document.getElementById('confidence');
    const confidenceValue = document.getElementById('confidence-value');
    if (confidenceSlider && confidenceValue) {
        confidenceSlider.addEventListener('input', function() {
            confidenceValue.textContent = this.value + '%';
            // Update detector confidence threshold
            if (window.emotionDetector && window.emotionDetector.detector) {
                window.emotionDetector.detector.confidence_threshold = this.value / 100;
            }
        });
    }
    
    // Setup reset tracking button
    const resetButton = document.getElementById('reset-tracking');
    if (resetButton) {
        resetButton.addEventListener('click', function() {
            resetTracking();
        });
    }
    
    // Setup fullscreen toggle
    const fullscreenButton = document.getElementById('fullscreen-toggle');
    if (fullscreenButton) {
        fullscreenButton.addEventListener('click', function() {
            toggleFullscreen();
        });
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

// Reset tracking data
function resetTracking() {
    console.log('🔄 Resetting tracking data...');
    
    // Reset emotion history
    emotionHistory = {};
    
    // Reset charts
    updateEmotionChart([]);
    updateTemporalChart([]); // Clear temporal chart
    
    // Reset statistics
    updateStatsDisplay({
        total_faces: 0,
        processing_time: 0,
        dominant_emotion: 'None',
        emotion_stability: 0
    });
    
    // Clear canvas
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    // Update storage info
    updateStorageInfo();
    
    console.log('✅ Tracking data reset successfully');
}

// Toggle fullscreen function
function toggleFullscreen() {
    const videoContainer = document.querySelector('.video-container');
    
    if (!document.fullscreenElement) {
        // Enter fullscreen
        if (videoContainer.requestFullscreen) {
            videoContainer.requestFullscreen();
        } else if (videoContainer.webkitRequestFullscreen) {
            videoContainer.webkitRequestFullscreen();
        } else if (videoContainer.msRequestFullscreen) {
            videoContainer.msRequestFullscreen();
        }
        showNotification('Entered fullscreen mode! 🔍');
    } else {
        // Exit fullscreen
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
        showNotification('Exited fullscreen mode! 📱');
    }
} 

// Reset emotion bias
function resetEmotionBias() {
    console.log('🔄 Resetting emotion bias...');
    
    fetch('/api/reset-emotion-bias', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('✅ Emotion bias reset successfully');
            updateEmotionStats();
            updateStorageInfo();
            showNotification('Emotion bias reset! 🔄');
        } else {
            console.error('❌ Error resetting emotion bias:', data.error);
            showNotification('Error resetting bias! ❌');
        }
    })
    .catch(error => {
        console.error('❌ Error resetting emotion bias:', error);
        showNotification('Error resetting bias! ❌');
    });
}

// Balance emotion weights
function balanceEmotionWeights() {
    console.log('⚖️ Balancing emotion weights...');
    
    const targetDistribution = {
        'Happy': 0.20,
        'Neutral': 0.20,
        'Sad': 0.20,
        'Angry': 0.20,
        'Surprised': 0.20
    };
    
    fetch('/api/adjust-emotion-weights', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ target_distribution: targetDistribution })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('✅ Emotion weights balanced successfully');
            showNotification('Emotion weights balanced! ⚖️');
        } else {
            console.error('❌ Error balancing emotion weights:', data.error);
            showNotification('Error balancing weights! ❌');
        }
    })
    .catch(error => {
        console.error('❌ Error balancing emotion weights:', error);
        showNotification('Error balancing weights! ❌');
    });
} 
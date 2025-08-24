let stream = null;
let lastCapturedFile = null;

async function toggleWebcam() {
    const webcamToggle = document.getElementById('webcam-toggle');
    const webcamOff = document.getElementById('webcam-off');
    const loading = document.getElementById('loading');
    const video = document.getElementById('webcam');
    const captureButton = document.getElementById('capture-button');
    const resultDiv = document.getElementById('capture-result');
    const detectionResultsDiv = document.getElementById('detection-results');

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        webcamOff.style.display = 'block';
        video.style.display = 'none';
        captureButton.disabled = true;
        webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn On Webcam';
        resultDiv.innerHTML = '';
        detectionResultsDiv.innerHTML = '';
        lastCapturedFile = null;
        document.getElementById('download-button-container').style.display = 'none';
    } else {
        webcamOff.style.display = 'none';
        loading.style.display = 'block';
        webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn Off Webcam';

        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                loading.style.display = 'none';
                video.style.display = 'block';
                captureButton.disabled = false;
                showNotification();
            };
        } catch (error) {
            loading.style.display = 'none';
            webcamOff.style.display = 'block';
            webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn On Webcam';
            resultDiv.innerHTML = `<p class="error-message">Cannot access webcam: ${error.message} 🚫</p>`;
        }
    }
}

async function capturePhoto() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('capture-canvas');
    const resultDiv = document.getElementById('capture-result');
    const detectionResultsDiv = document.getElementById('detection-results');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing photo...</p></div>';

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 1.0));
    const file = new File([blob], `capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
    lastCapturedFile = file;

    await processPhoto(file, resultDiv, detectionResultsDiv);
}

async function processPhoto(file, resultDiv, detectionResultsDiv) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', document.getElementById('confidence').value / 100);

    try {
        const response = await fetch('/capture', { method: 'POST', body: formData });
        if (!response.ok) {
            resultDiv.innerHTML = `<p class="error-message">Error processing photo: ${response.statusText} 🚫</p>`;
            detectionResultsDiv.innerHTML = '';
            return;
        }

        const result = await response.json();
        resultDiv.innerHTML = '';
        detectionResultsDiv.innerHTML = '';

        if (result.error) {
            resultDiv.innerHTML = `<p class="error-message">${result.error} 🚫</p>`;
            return;
        }

        if (result.image) {
            resultDiv.innerHTML = `
                <div class="image-container">
                    <img src="${result.image}" alt="Captured Photo" class="result-image">
                </div>`;
            const downloadButtonContainer = document.getElementById('download-button-container');
            downloadButtonContainer.style.display = 'block';
            const downloadButton = document.getElementById('download-image');
            downloadButton.onclick = () => {
                const a = document.createElement('a');
                a.href = result.image;
                a.download = `processed_capture_${Date.now()}.jpg`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };
        }

        if (result.results && result.detections && result.detections.length > 0) {
            detectionResultsDiv.innerHTML = '<h3>Detection Results 📊</h3>';
            result.detections.forEach(detection => {
                const emotion = detection.emotion || 'Unknown';
                const confidence = detection.emotion_confidence || 0;
                const confidencePercent = (confidence * 100).toFixed(1);
                const emoji = getEmotionEmoji(emotion);
                
                detectionResultsDiv.innerHTML += `
                    <div class="result-item ${emotion.toLowerCase()}">
                        <div class="result-header">
                            <span class="emotion-emoji">${emoji}</span>
                            <span class="emotion-label">${emotion}</span>
                            <span class="confidence-text">${confidencePercent}%</span>
                        </div>
                        <div class="result-details">
                            <span class="track-id">ID: ${detection.track_id}</span>
                            <span class="bbox-info">BBox: [${detection.bbox.join(', ')}]</span>
                        </div>
                    </div>`;
            });
        } else {
            detectionResultsDiv.innerHTML = '<p class="no-face">No faces detected! 😔</p>';
        }
        
        // Show processing time
        if (result.processing_time) {
            detectionResultsDiv.innerHTML += `
                <div class="processing-info">
                    <p>Processing time: ${(result.processing_time * 1000).toFixed(1)}ms</p>
                    <p>Total faces detected: ${result.total_faces || 0}</p>
                </div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="error-message">Error processing photo: ${error.message} 🚫</p>`;
        detectionResultsDiv.innerHTML = '';
    }
}

function getEmotionEmoji(emotion) {
    const emojiMap = {
        'Happy': '😊',
        'Neutral': '😐',
        'Sad': '😢',
        'Angry': '😠',
        'Surprised': '😲',
        'Unknown': '❓'
    };
    return emojiMap[emotion] || '❓';
}

function showNotification() {
    const notification = document.getElementById('notification');
    notification.style.display = 'flex';
    notification.classList.add('show');
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.style.display = 'none';
        }, 300);
    }, 3000);
}

document.getElementById('webcam-toggle').addEventListener('click', toggleWebcam);
document.getElementById('capture-button').addEventListener('click', capturePhoto);

document.getElementById('confidence').addEventListener('input', e => {
    document.getElementById('confidence-value').textContent = `${e.target.value}%`;
    e.target.style.background = `linear-gradient(to right, #d8b4fe ${e.target.value}%, #e0e0e0 ${e.target.value}%)`;
    if (lastCapturedFile) {
        const resultDiv = document.getElementById('capture-result');
        const detectionResultsDiv = document.getElementById('detection-results');
        resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing photo...</p></div>';
        processPhoto(lastCapturedFile, resultDiv, detectionResultsDiv);
    }
});

document.addEventListener('DOMContentLoaded', () => {
    showNotification();
});
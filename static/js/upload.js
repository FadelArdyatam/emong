// File: static/js/upload.js
let lastUploadedFile = null;

async function uploadFile() {
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const resultDiv = document.getElementById('upload-result');
    const detectionResultsDiv = document.getElementById('detection-results');
    const uploadButton = document.getElementById('upload-button');

    if (!fileInput.files[0]) {
        resultDiv.innerHTML = '<p class="error-message">Please select an image! 🚫</p>';
        return;
    }

    lastUploadedFile = fileInput.files[0];
    fileName.textContent = lastUploadedFile.name;

    resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing image...</p></div>';
    uploadButton.disabled = true;

    await processFile(lastUploadedFile, resultDiv, detectionResultsDiv, uploadButton);
}

async function processFile(file, resultDiv, detectionResultsDiv, uploadButton) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', document.getElementById('confidence').value / 100);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        if (!response.ok) {
            resultDiv.innerHTML = `<p class="error-message">Error processing image: ${response.statusText} 🚫</p>`;
            detectionResultsDiv.innerHTML = '';
            uploadButton.disabled = false;
            return;
        }

        const result = await response.json();
        resultDiv.innerHTML = '';
        detectionResultsDiv.innerHTML = '';

        if (result.error) {
            resultDiv.innerHTML = `<p class="error-message">${result.error} 🚫</p>`;
            uploadButton.disabled = false;
            return;
        }

        if (result.original_image && result.processed_image) {
            resultDiv.innerHTML = `
                <div class="image-comparison">
                    <div class="image-container">
                        <h4>Original Image</h4>
                        <img src="${result.original_image}" alt="Original Image" class="result-image">
                    </div>
                    <div class="image-container">
                        <h4>Processed Image</h4>
                        <img src="${result.processed_image}" alt="Processed Image" class="result-image">
                    </div>
                </div>`;
            document.getElementById('download-button-container').style.display = 'block';
            const downloadButton = document.getElementById('download-image');
            downloadButton.onclick = () => {
                const a = document.createElement('a');
                a.href = result.processed_image;
                a.download = `processed_${file.name}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };
        } else if (result.image) {
            resultDiv.innerHTML = `
                <div class="image-container">
                    <img src="${result.image}" alt="Uploaded Image" class="result-image">
                </div>`;
            document.getElementById('download-button-container').style.display = 'block';
            const downloadButton = document.getElementById('download-image');
            downloadButton.onclick = () => {
                const a = document.createElement('a');
                a.href = result.image;
                a.download = `processed_image_${Date.now()}.jpg`;
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
        resultDiv.innerHTML = `<p class="error-message">Error processing image: ${error.message} 🚫</p>`;
        detectionResultsDiv.innerHTML = '';
    }
    uploadButton.disabled = false;
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

function downloadResults(results, thumbnail) {
    const resultText = results.map(r => `${r.emotion}: ${r.confidence * 100}%`).join('\n');
    const blob = new Blob([`Detection Results:\n${resultText}\n\nThumbnail URL: ${thumbnail || 'N/A'}`], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emotion_detection_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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

document.getElementById('upload-button').addEventListener('click', () => {
    document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', uploadFile);

document.getElementById('confidence').addEventListener('input', e => {
    document.getElementById('confidence-value').textContent = `${e.target.value}%`;
    e.target.style.background = `linear-gradient(to right, #d8b4fe ${e.target.value}%, #e0e0e0 ${e.target.value}%)`;
    if (lastUploadedFile) {
        const resultDiv = document.getElementById('upload-result');
        const detectionResultsDiv = document.getElementById('detection-results');
        const uploadButton = document.getElementById('upload-button');
        resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing image...</p></div>';
        uploadButton.disabled = true;
        processFile(lastUploadedFile, resultDiv, detectionResultsDiv, uploadButton);
    }
});

document.addEventListener('DOMContentLoaded', () => {
    showNotification();
});
let stream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let lastRecordedFile = null;

async function toggleWebcam() {
    const webcamToggle = document.getElementById('webcam-toggle');
    const webcamOff = document.getElementById('webcam-off');
    const loading = document.getElementById('loading');
    const video = document.getElementById('webcam');
    const recordButton = document.getElementById('record-button');
    const resultDiv = document.getElementById('record-result');
    const detectionResultsDiv = document.getElementById('detection-results');

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        webcamOff.style.display = 'block';
        video.style.display = 'none';
        recordButton.disabled = true;
        webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn On Webcam';
        if (isRecording) stopRecording();
        resultDiv.innerHTML = '';
        detectionResultsDiv.innerHTML = '';
        lastRecordedFile = null;
    } else {
        webcamOff.style.display = 'none';
        loading.style.display = 'block';
        webcamToggle.innerHTML = '<i class="fas fa-camera"></i> Turn Off Webcam';

        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                loading.style.display = 'none';
                video.style.display = 'block';
                recordButton.disabled = false;
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

function startRecording() {
    if (!stream) return;
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp8' });
    mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const file = new File([blob], `record_${Date.now()}.webm`, { type: 'video/webm' });
        lastRecordedFile = file;
        recordedChunks = [];
        await processVideo(file);
    };
    mediaRecorder.start();
    isRecording = true;
    document.getElementById('record-button').innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
    document.getElementById('detection-results').innerHTML = '';
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById('record-button').innerHTML = '<i class="fas fa-circle"></i> Start Recording';
    }
}

async function processVideo(videoFile) {
    const resultDiv = document.getElementById('record-result');
    const detectionResultsDiv = document.getElementById('detection-results');

    // Hiển thị animation loading
    resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing video...</p></div>';
    detectionResultsDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('confidence', document.getElementById('confidence').value / 100);

    try {
        const response = await fetch('/record', { method: 'POST', body: formData });
        if (!response.ok) {
            resultDiv.innerHTML = `<p class="error-message">Error processing video: ${response.statusText} 🚫</p>`;
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

        if (result.video) {
            resultDiv.innerHTML = `
                <div class="video-container">
                    <video id="recorded-video" controls autoplay>
                        <source src="${result.video}" type="video/webm">
                        Your browser does not support the video tag.
                    </video>
                </div>`;
            const videoElement = document.getElementById('recorded-video');
            videoElement.onloadedmetadata = () => {
                videoElement.play();
            };
        }

        if (result.results && result.results.length > 0) {
            detectionResultsDiv.innerHTML = '<h3>Detection Results 📊</h3>';
            result.results.forEach(r => {
                const confidencePercent = (r.confidence * 100).toFixed(0);
                detectionResultsDiv.innerHTML += `
                    <div class="result-item">
                        <span class="emotion-label">${r.emotion} ${r.emoji}</span>
                        <span class="confidence-text">(${confidencePercent}%)</span>
                    </div>`;
            });
        } else {
            detectionResultsDiv.innerHTML = '<p class="no-face">No faces detected!</p>';
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="error-message">Error processing video: ${error.message} 🚫</p>`;
        detectionResultsDiv.innerHTML = '';
    }
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
document.getElementById('record-button').addEventListener('click', () => {
    if (isRecording) stopRecording();
    else startRecording();
});

document.getElementById('confidence').addEventListener('input', e => {
    document.getElementById('confidence-value').textContent = `${e.target.value}%`;
    e.target.style.background = `linear-gradient(to right, #d8b4fe ${e.target.value}%, #e0e0e0 ${e.target.value}%)`;
    if (lastRecordedFile) {
        const resultDiv = document.getElementById('record-result');
        const detectionResultsDiv = document.getElementById('detection-results');
        resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Processing video...</p></div>';
        processVideo(lastRecordedFile);
    }
});

document.addEventListener('DOMContentLoaded', () => {
    showNotification();
});
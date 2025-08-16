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

        if (result.image) {
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
        resultDiv.innerHTML = `<p class="error-message">Error processing image!</p>`;
        detectionResultsDiv.innerHTML = '';
    }
    uploadButton.disabled = false;
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
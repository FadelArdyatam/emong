let socket = io();
let prevTime = 0;
let currentEmotion = 'Neutral';
let dominantEmotion = 'Neutral';
let emotionBuffer = [];
const EMOTION_BUFFER_SIZE = 5;
let isSendingFrames = false;
let lastFrameTime = 0;
const FRAME_INTERVAL = 100;
let initialEmotionBuffer = [];
let initialEmotionTimer = null;
let hasInitiatedChat = false;
let usedResponses = {};
let detectedLanguage = 'vi'; // Mặc định là tiếng Việt

socket.on('connect', () => {
    console.log('Socket.io connected successfully');
});

socket.on('disconnect', () => {
    console.log('Socket.io disconnected');
});

function getMostFrequentEmotion(emotions) {
    if (!emotions || emotions.length === 0) return 'Neutral';
    const emotionCount = {};
    emotions.forEach(emotion => {
        emotionCount[emotion] = (emotionCount[emotion] || 0) + 1;
    });
    return Object.keys(emotionCount).reduce((a, b) => emotionCount[a] > emotionCount[b] ? a : b);
}

function smoothEmotion(newEmotion) {
    if (!isSendingFrames) return currentEmotion;
    if (!newEmotion || newEmotion === 'No face detected') return currentEmotion;
    emotionBuffer.push(newEmotion);
    if (emotionBuffer.length > EMOTION_BUFFER_SIZE) {
        emotionBuffer.shift();
    }
    return getMostFrequentEmotion(emotionBuffer);
}

const emotionResponses = {
    Happy: [
        "Cậu trông vui quá trời luôn! 😄 Có gì thú vị đang xảy ra hả?",
        "Ôi, nụ cười của cậu làm tớ cũng vui lây nè! 😊 Kể tớ nghe đi, hôm nay có gì hot?",
        "Hôm nay cậu rạng rỡ ghê, chắc có tin tốt đúng không? 😄 Chia sẻ với tớ nào!"
    ],
    Sad: [
        "Ôi, trông cậu hơi buồn nè... 😔 Có gì tâm sự được không, tớ nghe đây!",
        "Cậu ơi, có chuyện gì làm cậu xuống mood vậy? 😢 Nói với tớ, biết đâu tớ giúp được!",
        "Nhìn cậu buồn tớ cũng xót lắm... 😔 Muốn chia sẻ gì với tớ không nè?"
    ],
    Angry: [
        "Ủa, cậu đang bực mình gì à? 😣 Kể tớ nghe, xả stress chút nào!",
        "Cậu trông hơi căng thẳng nè, có ai chọc giận cậu hả? 😤 Nói tớ nghe đi!",
        "Hình như cậu đang nóng trong người đúng không? 😣 Bình tĩnh, tâm sự với tớ nè!"
    ],
    Surprised: [
        "Haha, cậu bị bất ngờ gì mà mắt tròn xoe vậy? 😲 Kể tớ nghe với!",
        "Ủa, chuyện gì làm cậu ngạc nhiên thế? 😳 Có gì hot hông, chia sẻ nào!",
        "Nhìn cậu shock thế này chắc có drama gì đúng không? 😲 Nói tớ nghe nè!"
    ],
    Neutral: [
        "Cậu trông bình thản ghê, hôm nay thế nào rồi? 😊 Có gì kể tớ không?",
        "Hình như cậu đang chill đúng không? 😎 Kể tớ nghe hôm nay cậu làm gì nè!",
        "Cậu ơi, trông cậu thư giãn quá, có gì hay ho đang xảy ra không? 😊",
        "Nhìn cậu bình yên thế này, chắc ngày hôm nay ổn áp đúng không? 😄 Kể tớ nghe nào!"
    ],
    Fear: [
        "Ôi, cậu trông hơi lo lắng nè... 😟 Có gì đáng sợ hả, kể tớ nghe nào!",
        "Cậu ơi, sao trông cậu bất an thế? 😨 Tâm sự với tớ đi, tớ ở đây nè!",
        "Hình như cậu đang sợ gì đúng không? 😟 Nói với tớ, biết đâu tớ an ủi được!"
    ],
    Disgust: [
        "Ủa, cậu vừa thấy gì mà mặt nhăn vậy? 😖 Có gì kỳ cục hả, kể tớ nghe!",
        "Haha, nhìn cậu ghê tởm gì thế? 😝 Chuyện gì làm cậu phản ứng mạnh vậy?",
        "Cậu trông khó chịu ghê, có gì không ổn hả? 😖 Chia sẻ với tớ nè!"
    ],
    Contempt: [
        "Ôi, cậu nhìn kiểu khinh khinh thế này là có chuyện gì hả? 😏 Kể tớ nghe nào!",
        "Cậu ơi, sao trông cậu như đang chê ai đó vậy? 😆 Có drama gì không, chia sẻ đi!",
        "Haha, mặt cậu kiểu 'thật luôn á' đúng không? 😄 Chuyện gì khiến cậu thế này nè?"
    ]
};

const emotionResponsesEnglish = {
    Happy: [
        "OMG bro, u look so happy! 😍 What’s making u smile like that? Spill the tea! 🎉",
        "Ayy bro, ur smile is giving me life! 😊 What’s up, anything fun happen? 🌟",
        "Yo bro, u look super lit today! 😄 Got some good vibes to share? 🫶"
    ],
    Sad: [
        "Aww bro, u look so down... 🥺 What’s wrong? I’m here for u, let’s talk! 💖",
        "Hey bro, u okay? 😢 U seem kinda sad, wanna tell me what’s up? 🤗",
        "Oh no bro, u look so sad... 🥺 I gotchu, tell me what’s making u feel like this! 💙"
    ],
    Angry: [
        "Whoa bro, u look kinda pissed! 😤 What’s got u so mad? Tell me, I gotchu! 🤗",
        "Yo bro, u seem super annoyed! 😣 What’s making u so angry? Let’s chat! 🫶",
        "Hey bro, u look like u wanna punch smth! 😠 What’s up, spill it! 😤"
    ],
    Surprised: [
        "Whoa bro, u look so shocked! 😲 What’s got u like that? Tell me quick! 🎉",
        "OMG bro, ur face is like 😳! What happened, tell me everything! 🌟",
        "Ayy bro, u look super surprised! 😲 What’s the tea, spill it! 🫶"
    ],
    Neutral: [
        "Hey bro, u seem chill today! 😎 How’s ur day going? Got any fun stuff to share? 🌟",
        "Yo bro, u look pretty chill! 😊 How’s ur day, anything cool happen? 🫶",
        "Ayy bro, u seem relaxed! 😎 What’s up with u today, tell me! 🌟"
    ],
    Fear: [
        "Oh no bro, u look kinda scared! 😱 What’s freaking u out? I’m here, talk to me! 🤗",
        "Hey bro, u seem super spooked! 😨 What’s got u so scared? I gotchu! 💙",
        "Yo bro, u look like u saw a ghost! 😱 What’s scaring u, tell me! 🤗"
    ],
    Disgust: [
        "Eww bro, what’s making u look so grossed out? 🤢 Tell me, I wanna know! 😝",
        "Ayy bro, u look like u just saw smth nasty! 🤮 What’s up, spill it! 😝",
        "Yo bro, ur face is like ew! 🤢 What’s making u feel like that? Tell me! 😜"
    ],
    Contempt: [
        "Hmm, u look like u’re judging smth! 😏 What’s up? Spill it, I’m curious! 🤔",
        "Hey bro, u got that judgy look! 😆 What’s got u like that? Tell me! 🌟",
        "Yo bro, u look like u’re side-eyeing smth! 😏 What’s the tea, spill it! 🤔"
    ]
};

function getRandomResponse(emotion) {
    const responses = detectedLanguage === 'vi' ? emotionResponses[emotion] || emotionResponses.Neutral : emotionResponsesEnglish[emotion] || emotionResponsesEnglish.Neutral;
    if (!usedResponses[emotion]) usedResponses[emotion] = [];

    let availableResponses = responses.filter((_, idx) => !usedResponses[emotion].includes(idx));
    if (availableResponses.length === 0) {
        usedResponses[emotion] = [];
        availableResponses = responses;
    }

    const idx = Math.floor(Math.random() * availableResponses.length);
    const response = availableResponses[idx];
    const globalIdx = responses.indexOf(response);
    usedResponses[emotion].push(globalIdx);

    return response;
}

function sendFrame(video, confidence) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    function send() {
        if (!isSendingFrames) return;
        const now = performance.now();
        if (now - lastFrameTime < FRAME_INTERVAL) {
            requestAnimationFrame(send);
            return;
        }
        lastFrameTime = now;

        if (!video.srcObject || !video.videoWidth || !video.videoHeight) {
            isSendingFrames = false;
            return;
        }

        ctx.drawImage(video, 0, 0);
        canvas.toBlob(blob => {
            if (!isSendingFrames) return;
            blob.arrayBuffer().then(buffer => {
                const currentTime = performance.now();
                const fps = prevTime ? 1000 / (currentTime - prevTime) : 0;
                prevTime = currentTime;
                socket.emit('frame', {
                    image: new Uint8Array(buffer),
                    confidence: confidence,
                    fps: fps
                });
                document.getElementById('fps-value').textContent = fps.toFixed(1);
            });
        }, 'image/jpeg', 1.0);
        requestAnimationFrame(send);
    }
    isSendingFrames = true;
    lastFrameTime = performance.now();
    send();

    initialEmotionBuffer = [];
    hasInitiatedChat = false;
    if (initialEmotionTimer) clearTimeout(initialEmotionTimer);
    initialEmotionTimer = setTimeout(() => {
        dominantEmotion = getMostFrequentEmotion(initialEmotionBuffer);
        if (dominantEmotion !== 'No face detected' && !hasInitiatedChat) {
            const response = getRandomResponse(dominantEmotion);
            appendBotMessage(response);
            hasInitiatedChat = true;
            currentEmotion = dominantEmotion;
        }
    }, 5000);
}

socket.on('result_frame', data => {
    if (!isSendingFrames) return;
    const video = document.getElementById('realtime-video');
    const canvas = document.getElementById('realtime-canvas');
    const resultDiv = document.getElementById('realtime-result');
    const ctx = canvas.getContext('2d');

    if (!video.videoWidth || !video.videoHeight) return;

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    const canvasWidth = canvas.clientWidth;
    const canvasHeight = canvas.clientHeight;
    const scaleX = canvasWidth / videoWidth;
    const scaleY = canvasHeight / videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (data.error) {
        resultDiv.innerHTML = `<p style="color: red;">${data.error} 🚫</p>`;
        return;
    }

    data.results.forEach(r => {
        if (r.emotion !== 'No face detected' && r.bbox && r.bbox.length === 4) {
            let [x1, y1, x2, y2] = r.bbox;
            if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) return;

            x1 = x1 * scaleX;
            y1 = y1 * scaleY;
            x2 = x2 * scaleX;
            y2 = y2 * scaleY;

            x1 = Math.max(0, Math.min(x1, canvasWidth));
            y1 = Math.max(0, Math.min(y1, canvasHeight));
            x2 = Math.max(0, Math.min(x2, canvasWidth));
            y2 = Math.max(0, Math.min(y2, canvasHeight));

            ctx.strokeStyle = r.color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            const confidencePercent = (r.confidence * 100).toFixed(0);
            const label = `${r.emotion}: ${confidencePercent}%`;
            ctx.font = 'bold 18px Roboto';
            const textWidth = ctx.measureText(label).width;
            const textHeight = 18;

            let textX = x1;
            let textY = y1 - 10;
            let boxY = y1 - textHeight - 15;
            if (y1 - textHeight - 10 < 0) {
                textY = y2 + textHeight + 10;
                boxY = y2 + 5;
            }

            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            ctx.fillRect(x1, boxY, textWidth + 10, textHeight + 10);
            ctx.fillStyle = '#FFFFFF';
            ctx.fillText(label, x1 + 5, textY);
        }
    });

    const r = data.results[0] || {};
    const smoothedEmotion = smoothEmotion(r.emotion);
    const confidencePercent = (r.confidence * 100).toFixed(0);
    resultDiv.innerHTML = `
        <div class="result-container">
            Emotion: <span class="emotion-label">${smoothedEmotion} ${r.emoji || ''}</span> (Confidence: ${confidencePercent}%)
        </div>`;

    if (initialEmotionTimer && r.emotion && r.emotion !== 'No face detected') {
        initialEmotionBuffer.push(r.emotion);
    }
});

function appendBotMessage(message) {
    const chatDiv = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'bot-message';
    messageDiv.innerHTML = `
        <img src="/static/images/ic_launcher.png" alt="Bot Avatar" class="chat-avatar">
        <span>${message}</span>`;
    chatDiv.appendChild(messageDiv);
    chatDiv.scrollTop = chatDiv.scrollHeight;
}

socket.on('chat_responding', data => {
    const chatDiv = document.getElementById('chat-container');
    const respondingDiv = document.createElement('div');
    respondingDiv.className = 'responding-indicator';
    respondingDiv.textContent = '...';
    chatDiv.appendChild(respondingDiv);
    chatDiv.scrollTop = chatDiv.scrollHeight;
});

socket.on('chat_response', data => {
    const chatDiv = document.getElementById('chat-container');
    const statusDiv = document.getElementById('chatbot-status');
    chatDiv.querySelector('.responding-indicator')?.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = 'bot-message';
    messageDiv.innerHTML = `
        <img src="/static/images/ic_launcher.png" alt="Bot Avatar" class="chat-avatar">
        <span>${data.message}</span>`;
    chatDiv.appendChild(messageDiv);
    chatDiv.scrollTop = chatDiv.scrollHeight;

    statusDiv.className = `chatbot-status ${data.status}`;
    statusDiv.textContent = data.status === 'success'
        ? 'Chatbot is active! Response sent! 🤖✅'
        : 'Chatbot error... Please try again!';
});
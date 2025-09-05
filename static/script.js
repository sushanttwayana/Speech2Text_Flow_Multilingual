
let websocket;
let audioContext;
let mediaStreamSource;
let audioWorkletNode;
let audioStream;
let isRecording = false;
// Using crypto.randomUUID(), so clientId unique per session
const clientId = 'client-' + crypto.randomUUID();
const sampleRate = 16000; // Fixed sample rate for Whisper models (16kHz)


console.log("Hello I have entered!!!")
/**
 * Establishes a WebSocket connection to the backend.
 */
function connectWebSocket() {
    // const wsUrl = `ws://127.0.0.1:8000//ws/${clientId}`;
    // const wsUrl = `ws://127.0.0.1:8001/ws/${clientId}`;
    // const wsUrl = `ws://localhost:8000/ws/${clientId}`;
    // websocket = new WhostebSocket(wsUrl);

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host; 
    const wsUrl = `${wsProtocol}//${wsHost}/ws/${clientId}`;
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
        console.log('WebSocket connected to', wsUrl);
        updateStatus('Connected to server', 'ready');
    };
    
    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received message:', data);

        if (data.type === 'transcription') {
            if (data.is_final) {
                displayFinalTranscription(data.text);
            } else {
                displayInterimTranscription(data.text);
            }
        } else if (data.type === 'status') {
            updateStatus(data.message,
                data.message.includes('Recording') ? 'recording' :
                data.message.includes('Processing') ? 'processing' : 'ready');
        } else if (data.type === 'error') {
            console.error('Server Error:', data.message);
            updateStatus('Error: ' + data.message, 'error');
            if (isRecording) stopRecording();
        } else if (data.type === 'pong') {
            console.log('Pong received at', data.timestamp);
        } else if (data.type === 'navigate') {
            // ** Handle navigation messages from backend to redirect user **
            console.log('Handling navigate message:', data);

            // Construct query string from prefill data
            const params = new URLSearchParams();
            if (data.prefill) {
                for (const [key, value] of Object.entries(data.prefill)) {
                    if (value !== undefined && value !== null && value !== "") {
                        params.append(key, value);
                        console.log(`Prefilling ${key}: ${value}`);
                    }
                }
            }
            
            // Construct full URL with query params
            const url = data.route + (params.toString() ? '?' + params.toString() : '');
            console.log('Navigating to:', url);

            // Redirect browser to the URL with prefilled query params
            window.location.href = url;
        } else {
            console.warn('Unknown message type:', data.type);
        }
    };


    websocket.onclose = () => {
        console.log('WebSocket closed');
        updateStatus('Disconnected from server', 'error');
        if (isRecording) handleRecordingStopped();
    };


    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('WebSocket Error', 'error');
        if (isRecording) stopRecording();
    };
}

/**
 * Starts the audio recording and streaming to the WebSocket.
 */
async function startRecording() {
    try {
        console.log('Starting recording...');
        // Clear previous transcriptions on new recording session
        document.getElementById('interimTranscription').textContent = '';
        document.getElementById('finalTranscription').textContent = 'Listening for commands...';
        
        // Ensure WebSocket is connected before starting audio capture
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            console.log('WebSocket not open, attempting to connect...');
            connectWebSocket();
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error('WebSocket connection timed out.')), 5000);
                websocket.onopen = () => { clearTimeout(timeout); resolve(); };
                websocket.onerror = (e) => { clearTimeout(timeout); reject(e); };
            });
        }
        
        // Request microphone access from the user
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: sampleRate,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            },
            video: false
        }).catch(err => {
            console.error('getUserMedia error:', err.name, err.message);
            throw new Error('Microphone access denied: ' + err.message);
        });


        // Create AudioContext for processing audio stream
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
        mediaStreamSource = audioContext.createMediaStreamSource(audioStream);
        
        // Define AudioWorklet processor code
        const audioWorkletCode = `
            class AudioProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                }
                process(inputs, outputs, parameters) {
                    const input = inputs[0];
                    if (input.length > 0) {
                        const audioData = input[0];
                        const audioInt16 = new Int16Array(audioData.length);
                        for (let i = 0; i < audioData.length; i++) {
                            audioInt16[i] = Math.min(32767, Math.max(-32768, audioData[i] * 32768));
                        }
                        this.port.postMessage(audioInt16.buffer, [audioInt16.buffer]);
                    }
                    return true;
                }
            }
            registerProcessor('audio-processor', AudioProcessor);
        `;
        
        // Add the module to the audio context
        const blob = new Blob([audioWorkletCode], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        await audioContext.audioWorklet.addModule(url);
        
        // Create AudioWorkletNode
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        
        audioWorkletNode.port.onmessage = (event) => {
            if (!isRecording) return;
            const audioBuffer = event.data;
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'audio_chunk',
                    audio: arrayBufferToBase64(audioBuffer)
                }));
            }
        };
        
        // Connect nodes: source -> worklet -> destination
        mediaStreamSource.connect(audioWorkletNode);
        audioWorkletNode.connect(audioContext.destination);

        // Inform the backend that recording has started
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
                type: 'start_recording',
                sample_rate: sampleRate
            }));
            console.log('Sent start_recording message');
        }
        
        isRecording = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        updateStatus('Recording... Speak your command!', 'recording');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        updateStatus('Error: Could not access microphone. Please check permissions.', 'error');
        handleRecordingStopped();
    }
}

/**
 * Stops the audio recording and sends a stop signal to the backend.
 */
function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: 'stop_recording'
        }));
        console.log('Sent stop_recording message');
    }
    
    handleRecordingStopped();
    updateStatus('Processing final command...', 'processing');
}

/**
 * Cleans up local audio recording resources (microphone, audio context).
 */
function handleRecordingStopped() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
    if (audioWorkletNode) {
        audioWorkletNode.disconnect();
        audioWorkletNode = null;
    }
    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
        mediaStreamSource = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
        audioContext = null;
    }
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

/**
 * Converts an ArrayBuffer to a Base64 string.
 * @param {ArrayBuffer} buffer The ArrayBuffer to convert.
 * @returns {string} The Base64 encoded string.
 */
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

/**
 * Updates the status message displayed on the page.
 * @param {string} message The message to display.
 * @param {string} type A class name (e.g., 'ready', 'recording', 'error') for styling.
 */
function updateStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = 'status ' + type;
}

/**
 * Displays interim transcription results.
 * @param {string} text The interim transcription text.
 */
function displayInterimTranscription(text) {
    const interimDiv = document.getElementById('interimTranscription');
    interimDiv.textContent = text;
}

/**
 * Displays the final transcription result and clears interim.
 * @param {string} text The final transcription text.
 */
function displayFinalTranscription(text) {
    const finalDiv = document.getElementById('finalTranscription');
    finalDiv.textContent = text;
    document.getElementById('interimTranscription').textContent = '';
}

// Initialize UI and WebSocket connection when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    console.log('Buttons found:', startBtn, stopBtn);
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            console.log('Start button clicked');
            startRecording();
        });
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopRecording);
    }
    connectWebSocket();
});

// Clean up resources when the page is unloaded
window.addEventListener('beforeunload', () => {
    if (isRecording) stopRecording();
    if (websocket && websocket.readyState === WebSocket.OPEN) websocket.close();
});
class SecurityCamera {
    constructor() {
        this.stream = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.videoUrl = null;
        this.ws = null;
        this.isConnected = false;
        this.videoStarted = false;
        this.initializeElements();
        this.addEventListeners();
        this.setupCanvas();
    }

    initializeElements() {
        this.ipInput = document.getElementById('ipAddress');
        this.connectBtn = document.getElementById('connectBtn');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.recordBtn = document.getElementById('recordBtn');
        this.video = document.getElementById('cameraFeed');
        this.canvas = document.getElementById('overlay');
        this.status = document.getElementById('status');
        this.alertList = document.getElementById('alertList');
        this.ctx = this.canvas.getContext('2d');
        this.videoFile = document.getElementById('videoFile');
        this.loadBtn = document.getElementById('loadBtn');
        this.videoProgress = document.getElementById('videoProgress');
        this.timestamp = document.getElementById('timestamp');
    }

    addEventListeners() {
        this.connectBtn.addEventListener('click', () => this.connectCamera());
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.captureBtn.addEventListener('click', () => this.takeScreenshot());
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.loadBtn.addEventListener('click', () => this.loadVideo());
        this.videoProgress.addEventListener('input', () => this.seekVideo());
        this.video.addEventListener('timeupdate', () => this.updateProgress());
        this.startBtn.addEventListener('click', () => this.startVideo());
        this.stopBtn.addEventListener('click', () => this.stopVideo());
    }

    setupCanvas() {
        // Set initial canvas size
        this.canvas.width = 960;
        this.canvas.height = 540;
        // Lấy context với willReadFrequently
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        
        // Style for video element
        this.video.style.width = '960px';
        this.video.style.height = '540px';
        this.video.style.objectFit = 'contain';
        this.video.style.display = 'block';

        // Đặt thuộc tính cho video
        this.video.setAttribute('autoplay', '');
        this.video.setAttribute('playsinline', '');
        this.video.setAttribute('muted', '');
    }

    async connectCamera() {
        try {
            this.ws = new WebSocket('ws://localhost:8000/ws');
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.videoStarted = false;
                this.status.textContent = 'Connected to YOLO Server';
                this.status.style.backgroundColor = 'rgba(0,255,0,0.7)';
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.error) {
                        console.error('Server error:', data.error);
                        this.status.textContent = 'Error: ' + data.error;
                        this.status.style.backgroundColor = 'rgba(255,0,0,0.7)';
                        return;
                    }

                    // Xử lý thông báo khẩn cấp nếu có
                    if (data.emergency_alerts && data.emergency_alerts.length > 0) {
                        data.emergency_alerts.forEach(alert => {
                            this.showEmergencyAlert(alert);
                        });
                    }

                    const img = new Image();
                    img.onload = () => {
                        // Chỉ vẽ frame video, bỏ qua phần detection
                        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                        this.ctx.drawImage(img, 0, 0);
                    };
                    img.src = 'data:image/jpeg;base64,' + data.image;
                } catch (error) {
                    console.error('Error processing message:', error);
                }
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.videoStarted = false;
                this.status.textContent = 'Disconnected';
                this.status.style.backgroundColor = 'rgba(255,0,0,0.7)';
            };
            
        } catch (error) {
            console.error('Error connecting to server:', error);
            this.status.textContent = 'Connection Failed';
            this.status.style.backgroundColor = 'rgba(255,0,0,0.7)';
        }
    }

    startCamera() {
        if (this.isConnected) {
            this.status.textContent = 'Camera Running';
            this.status.style.backgroundColor = 'rgba(0,255,0,0.7)';
            this.addAlert('Camera started');
        }
    }

    takeScreenshot() {
        const canvas = document.createElement('canvas');
        canvas.width = this.video.videoWidth;
        canvas.height = this.video.videoHeight;
        canvas.getContext('2d').drawImage(this.video, 0, 0);
        
        // Create download link
        const link = document.createElement('a');
        link.download = `screenshot_${new Date().toISOString()}.png`;
        link.href = canvas.toDataURL();
        link.click();
    }

    toggleRecording() {
        if (!this.isRecording) {
            this.startRecording();
        } else {
            this.stopRecording();
        }
    }

    startRecording() {
        this.recordedChunks = [];
        const stream = this.video.captureStream();
        this.mediaRecorder = new MediaRecorder(stream);
        
        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                this.recordedChunks.push(e.data);
            }
        };

        this.mediaRecorder.onstop = () => {
            const blob = new Blob(this.recordedChunks, {type: 'video/webm'});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `recording_${new Date().toISOString()}.webm`;
            link.click();
        };

        this.mediaRecorder.start();
        this.isRecording = true;
        this.recordBtn.textContent = 'Stop Recording';
        this.addAlert('Recording started');
    }

    stopRecording() {
        this.mediaRecorder.stop();
        this.isRecording = false;
        this.recordBtn.textContent = 'Start Recording';
        this.addAlert('Recording stopped');
    }

    loadVideo() {
        const file = this.videoFile.files[0];
        if (file) {
            if (this.videoUrl) {
                URL.revokeObjectURL(this.videoUrl);
            }
            this.videoUrl = URL.createObjectURL(file);
            this.video.src = this.videoUrl;
            this.status.textContent = 'Video Loaded';
            this.status.style.backgroundColor = 'rgba(0,255,0,0.7)';
            
            // Reset progress
            this.videoProgress.value = 0;
            this.updateTimestamp();
            
            // Start detection when video is loaded
            this.video.onloadedmetadata = () => {
                this.startDetection();
            };
        }
    }

    startVideo() {
        if (this.video.paused && this.video.src) {
            this.video.play();
            this.status.textContent = 'Playing';
        }
    }

    stopVideo() {
        if (!this.video.paused && this.video.src) {
            this.video.pause();
            this.status.textContent = 'Paused';
        }
    }

    seekVideo() {
        if (this.video.src) {
            const time = (this.videoProgress.value / 100) * this.video.duration;
            this.video.currentTime = time;
        }
    }

    updateProgress() {
        if (this.video.src) {
            const progress = (this.video.currentTime / this.video.duration) * 100;
            this.videoProgress.value = progress;
            this.updateTimestamp();
        }
    }

    updateTimestamp() {
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };

        const current = formatTime(this.video.currentTime);
        const total = formatTime(this.video.duration || 0);
        this.timestamp.textContent = `${current} / ${total}`;
    }

    startDetection() {
        const detectFrame = () => {
            if (!this.video.paused && !this.video.ended && this.video.src) {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Simulated detection - replace with actual YOLO detection
                if (Math.random() < 0.1) { // 10% chance of detection
                    this.drawDetection();
                    this.addAlert('Motion detected!');
                }
            }
            requestAnimationFrame(detectFrame);
        };
        
        detectFrame();
    }

    drawDetection() {
        this.ctx.strokeStyle = 'red';
        this.ctx.lineWidth = 2;
        const x = Math.random() * (this.canvas.width - 100);
        const y = Math.random() * (this.canvas.height - 100);
        this.ctx.strokeRect(x, y, 100, 100);
    }

    drawDetections(data) {
        const beds = data.beds || [];
        const pillows = data.pillows || [];
        const detections = data.detections || [];
        
        // Update canvas size if dimensions are provided
        if (data.dimensions) {
            if (this.canvas.width !== data.dimensions.width ||
                this.canvas.height !== data.dimensions.height) {
                this.canvas.width = data.dimensions.width;
                this.canvas.height = data.dimensions.height;
                this.video.style.width = `${data.dimensions.width}px`;
                this.video.style.height = `${data.dimensions.height}px`;
            }
        }
        
        // Clear only the overlay canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Set canvas context properties for better visibility
        this.ctx.lineWidth = 2;
        this.ctx.font = '12px Arial';
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        
        // Draw beds first (background)
        for (const bed of beds) {
            const [x1, y1, x2, y2] = bed;
            this.ctx.strokeStyle = 'rgb(0, 255, 255)';  // Yellow for beds
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, x2-x1, y2-y1);
            this.ctx.fillText('bed', x1, y1-5);
        }

        // Draw pillows
        for (const pillow of pillows) {
            const [x1, y1, x2, y2] = pillow;
            this.ctx.strokeStyle = 'rgb(255, 0, 255)';  // Magenta for pillows
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, x2-x1, y2-y1);
            this.ctx.fillText('pillow', x1, y1-5);
        }

        // Draw person detections
        for (const det of detections) {
            const {box, confidence, state, status, ratio} = det;
            const [x1, y1, x2, y2] = box;

            // Chọn màu dựa trên status
            let color = {
                "sleep": "rgb(255, 191, 0)",
                "fall": "rgb(0, 255, 255)",
                "fall_prepare": "rgb(0, 69, 255)",
                "fall_alert": "rgb(255, 0, 0)",
                "normal": "rgb(0, 255, 0)",
                "like_fall_1": "rgb(0, 255, 255)",
                "like_fall_2": "rgb(0, 200, 255)",
                "like_fall_3": "rgb(0, 140, 255)",
                "like_fall_4": "rgb(0, 69, 255)"
            }[status] || "rgb(0, 255, 0)";

            // Vẽ bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x1, y1, x2-x1, y2-y1);

            // Vẽ label
            this.ctx.fillStyle = color;
            let label = `Person K(${ratio.toFixed(2)}) - X(${confidence.toFixed(2)})`;
            if (state.includes('like_fall')) {
                const level = state.split('_').pop();
                label += ` - ${state.replace('_'+level, '')} (L${level})`;
            } else {
                label += ` - ${state}`;
            }
            if (status) {
                label += ` [${status}]`;
            }
            this.ctx.fillText(label, x1, y1-5);

            // Thêm cảnh báo nếu là fall_alert
            if (status === 'fall_alert') {
                this.addAlert('FALL ALERT - EMERGENCY!');
            }

            // Add emergency alert
            if (det.is_emergency) {
                this.ctx.fillStyle = 'rgb(255, 0, 0)';
                this.ctx.font = '24px Arial';
                this.ctx.fillText('EMERGENCY SIGNAL DETECTED!', 20, 80);
                this.addAlert('EMERGENCY SIGNAL DETECTED!');
            }
        }
    }

    addAlert(message) {
        const alertDiv = document.createElement('div');
        alertDiv.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        this.alertList.insertBefore(alertDiv, this.alertList.firstChild);
    }

    // Thêm phương thức mới để hiển thị thông báo khẩn cấp
    showEmergencyAlert(alert) {
        // Thêm vào alert list
        this.addAlert(alert.message);
        
        // Thay đổi màu status
        this.status.textContent = alert.message;
        this.status.style.backgroundColor = 'rgba(255,0,0,0.9)';
        
        // Phát âm thanh cảnh báo (tùy chọn)
        const audio = new Audio('/alert.mp3');
        audio.play().catch(e => console.log('Audio play failed:', e));
        
        // Flash effect trên UI (tùy chọn)
        this.status.style.animation = 'flash 1s infinite';
    }
}

// Thêm CSS animation cho flash effect
const style = document.createElement('style');
style.textContent = `
    @keyframes flash {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
`;
document.head.appendChild(style);

// Initialize the security camera system
window.addEventListener('load', () => {
    const securityCam = new SecurityCamera();
});

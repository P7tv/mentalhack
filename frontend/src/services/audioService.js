export async function recordAudio(duration = 10000) {
  return new Promise(async (resolve, reject) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        stream.getTracks().forEach(track => track.stop());
        resolve(audioBlob);
      };

      mediaRecorder.start();

      setTimeout(() => {
        mediaRecorder.stop();
      }, duration);
    } catch (error) {
      reject(error);
    }
  });
}

export async function speechToText(audioBlob) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    throw new Error('Speech Recognition ไม่รองรับในเบราวเซอร์นี้');
  }

  return new Promise((resolve, reject) => {
    const recognition = new SpeechRecognition();
    recognition.lang = 'th-TH';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      resolve(transcript);
    };

    recognition.onerror = (event) => {
      reject(new Error(`Speech Recognition Error: ${event.error}`));
    };

    recognition.onend = () => {
      if (!recognition.results || recognition.results.length === 0) {
        reject(new Error('ไม่สามารถจดจำเสียงได้'));
      }
    };

    const audioContext = new AudioContext();
    const reader = new FileReader();

    reader.onload = (e) => {
      audioContext.decodeAudioData(e.target.result, (audioBuffer) => {
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start(0);
        recognition.start();
      });
    };

    reader.readAsArrayBuffer(audioBlob);
  });
}

export async function recordAndTranscribe() {
  const audioBlob = await recordAudio();
  const transcript = await speechToText(audioBlob);
  return transcript;
}

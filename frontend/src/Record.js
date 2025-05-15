//Record.js


import React, { useState } from 'react';

function Record({ setResult }) {
  const [audioBlob, setAudioBlob] = useState(null);
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = React.useRef(null);
  const chunks = [];

  const startRecording = async () => {
    setRecording(true);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);

    mediaRecorderRef.current.ondataavailable = (e) => {
      chunks.push(e.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      setAudioBlob(blob);
    };

    mediaRecorderRef.current.start();
  };

  const stopRecording = () => {
    setRecording(false);
    mediaRecorderRef.current.stop();
  };

  const sendAudio = async () => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    const response = await fetch('http://localhost:8000/predict/', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    setResult(result);
  };

  return (
    <div>
      <button onClick={recording ? stopRecording : startRecording}>
        {recording ? 'Stop' : 'Record'}
      </button>
      {audioBlob && <button onClick={sendAudio}>Analyze</button>}
    </div>
);
}

export default Record;

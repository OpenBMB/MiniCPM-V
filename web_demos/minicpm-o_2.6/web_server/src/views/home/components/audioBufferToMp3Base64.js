import lame from '@breezystack/lamejs';

export const audioBufferToMp3Base64 = audioBuffer => {
    const mp3Encoder = new lame.Mp3Encoder(1, 16000, 128);
    const sampleBlockSize = 1152;
    const mp3Data = [];

    for (let i = 0; i < audioBuffer.length; i += sampleBlockSize) {
        const sampleChunk = audioBuffer.subarray(i, i + sampleBlockSize);
        const mp3buf = mp3Encoder.encodeBuffer(sampleChunk);
        if (mp3buf.length > 0) {
            mp3Data.push(new Int8Array(mp3buf));
        }
    }

    const mp3buf = mp3Encoder.flush();
    if (mp3buf.length > 0) {
        mp3Data.push(new Int8Array(mp3buf));
    }

    const mp3Blob = new Blob(mp3Data, { type: 'audio/mp3' });
    const url = URL.createObjectURL(mp3Blob);
    let dom = document.querySelector('#voice-box');
    let audio = document.createElement('audio');
    audio.controls = true;
    audio.src = url;
    dom.appendChild(audio);
    return new Promise(resolve => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result.split(',')[1];
            resolve(base64String);
        };
        reader.readAsDataURL(mp3Blob);
    });
};

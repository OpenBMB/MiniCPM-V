// Convert Base64 to ArrayBuffer
const base64ToArrayBuffer = base64 => {
    const binaryString = atob(base64.split(',')[1]); // Remove data URI scheme if present
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
};

// Parse WAV header and get audio data section
const parseWav = buffer => {
    const view = new DataView(buffer);
    const format = view.getUint16(20, true);
    const channels = view.getUint16(22, true);
    const sampleRate = view.getUint32(24, true);
    const bitsPerSample = view.getUint16(34, true);
    const dataOffset = 44;
    const dataSize = view.getUint32(40, true);
    const audioData = new Uint8Array(buffer, dataOffset, dataSize);

    return {
        format,
        channels,
        sampleRate,
        bitsPerSample,
        audioData
    };
};

// Create WAV header for combined audio data
const createWavHeader = (audioDataSize, sampleRate, channels, bitsPerSample) => {
    const arrayBuffer = new ArrayBuffer(44);
    const view = new DataView(arrayBuffer);

    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(view, 0, 'RIFF'); // ChunkID
    view.setUint32(4, 36 + audioDataSize, true); // ChunkSize
    writeString(view, 8, 'WAVE'); // Format
    writeString(view, 12, 'fmt '); // Subchunk1ID
    view.setUint32(16, 16, true); // Subchunk1Size (PCM)
    view.setUint16(20, 1, true); // AudioFormat (PCM)
    view.setUint16(22, channels, true); // NumChannels
    view.setUint32(24, sampleRate, true); // SampleRate
    view.setUint32(28, (sampleRate * channels * bitsPerSample) / 8, true); // ByteRate
    view.setUint16(32, (channels * bitsPerSample) / 8, true); // BlockAlign
    view.setUint16(34, bitsPerSample, true); // BitsPerSample
    writeString(view, 36, 'data'); // Subchunk2ID
    view.setUint32(40, audioDataSize, true); // Subchunk2Size

    return arrayBuffer;
};

// Merge multiple Base64 audio files and return a Blob
const mergeAudioFiles = base64AudioArray => {
    let sampleRate, channels, bitsPerSample;
    let combinedAudioData = new Uint8Array();

    for (let i = 0; i < base64AudioArray.length; i++) {
        const arrayBuffer = base64ToArrayBuffer(base64AudioArray[i]);
        const wav = parseWav(arrayBuffer);

        // Initialize properties based on the first audio file
        if (i === 0) {
            sampleRate = wav.sampleRate;
            channels = wav.channels;
            bitsPerSample = wav.bitsPerSample;
        }

        // Ensure all files have the same format
        if (wav.sampleRate !== sampleRate || wav.channels !== channels || wav.bitsPerSample !== bitsPerSample) {
            throw new Error('All audio files must have the same format.');
        }

        // Combine audio data
        const newCombinedData = new Uint8Array(combinedAudioData.byteLength + wav.audioData.byteLength);
        newCombinedData.set(combinedAudioData, 0);
        newCombinedData.set(wav.audioData, combinedAudioData.byteLength);
        combinedAudioData = newCombinedData;
    }

    const combinedAudioDataSize = combinedAudioData.byteLength;
    const wavHeader = createWavHeader(combinedAudioDataSize, sampleRate, channels, bitsPerSample);
    const combinedWavBuffer = new Uint8Array(wavHeader.byteLength + combinedAudioData.byteLength);
    combinedWavBuffer.set(new Uint8Array(wavHeader), 0);
    combinedWavBuffer.set(combinedAudioData, wavHeader.byteLength);

    // Create a Blob from the combined audio data
    const combinedBlob = new Blob([combinedWavBuffer], { type: 'audio/wav' });
    return combinedBlob;
};
export const mergeBase64ToBlob = base64List => {
    const combinedBlob = mergeAudioFiles(base64List);
    const audioUrl = URL.createObjectURL(combinedBlob);
    return audioUrl;
};

// 假设 base64Strings 是一个包含多个 Base64 编码 WAV 文件的数组
// 注意：这些 Base64 字符串不应该包含 URI 前缀 (例如 "audio/wav;base64,")
/**
 *
 * @param {Array} base64Strings
 * @returns
 */
// 解码 Base64 字符串并合并二进制数据
export const mergeBase64WavFiles = base64Strings => {
    const binaryDataArray = base64Strings.map(base64 => {
        return Uint8Array.from(atob(base64), c => c.charCodeAt(0));
    });

    const totalLength = binaryDataArray.reduce((sum, arr) => sum + arr.length, 0);

    const mergedArray = new Uint8Array(totalLength);
    let offset = 0;

    binaryDataArray.forEach(arr => {
        mergedArray.set(arr, offset);
        offset += arr.length;
    });

    // 重新编码为 Base64 字符串
    const binaryString = String.fromCharCode(...mergedArray);
    const mergedBase64 = btoa(binaryString);

    return mergedBase64;
};

const base64ToArrayBuffer = base64 => {
    let binaryString = atob(base64);
    let len = binaryString.length;
    let bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
};

const concatenateArrayBuffers = buffers => {
    let totalLength = buffers.reduce((acc, value) => acc + value.byteLength, 0);
    let result = new Uint8Array(totalLength);
    let offset = 0;
    for (let buffer of buffers) {
        result.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    }
    return result.buffer;
};

export const mergeMp3Base64ToBlob = base64Strings => {
    let arrayBuffers = base64Strings.map(base64ToArrayBuffer);
    let combinedArrayBuffer = concatenateArrayBuffers(arrayBuffers);
    const blob = new Blob([combinedArrayBuffer], { type: 'audio/mp3' });
    const url = URL.createObjectURL(blob);
    console.log('url', url);
    return url;
};

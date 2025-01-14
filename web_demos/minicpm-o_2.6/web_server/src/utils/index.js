// 判断终端是pc还是移动端
export const isMobile = () => {
    let flag = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Linux/i.test(navigator.userAgent);
    const platform = navigator.platform;
    // iPad上的Safari
    if (platform === 'MacIntel' && navigator.maxTouchPoints > 1) {
        flag = true;
    }
    return flag;
};
// 单片语音长度(单位：ms)
const voicePerLength = 200;

// 图片计数，算出在哪一次发送语音时，同时发送图片。例如一片语音100ms，一秒钟发送一次语音，即发送的第10片语音时需要带一张图片
export const maxCount = 1000 / voicePerLength;

export const getChunkLength = sampleRate => {
    return sampleRate * (voicePerLength / 1000);
};

export const isAvailablePort = port => {
    return [
        8000, 8001, 8002, 8003, 8004, 8010, 8011, 8012, 8013, 8014, 8020, 8021, 8022, 8023, 8024, 8025, 8026, 8027,
        8028, 32449
    ].includes(port);
};

// 文件转base64格式
export const fileToBase64 = file => {
    return new Promise((resolve, reject) => {
        if (!file) {
            reject('文件不能为空');
        }
        const reader = new FileReader();
        reader.onload = e => {
            const base64String = e.target.result;
            resolve(base64String);
        };
        reader.onerror = () => {
            reject('文件转码失败');
        };
        reader.readAsDataURL(file);
    });
};

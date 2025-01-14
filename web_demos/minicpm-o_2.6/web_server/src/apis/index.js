// 定时发送消息
export const sendMessage = data => {
    return useHttp.post('/api/v1/stream', data);
};
// 跳过当前
export const stopMessage = () => {
    return useHttp.post('/api/v1/stop');
};
// 上传音色文件
export const uploadFile = data => {
    return useHttp.post('/api/v1/upload_audio', data);
};
// 反馈
export const feedback = data => {
    return useHttp.post('/api/v1/feedback', data);
};
// 上传配置
export const uploadConfig = data => {
    return useHttp.post('/api/v1/init_options', data);
    // return useHttp.post('/api/v1/upload_audio', data);
};

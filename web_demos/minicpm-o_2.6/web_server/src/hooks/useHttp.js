import axios from 'axios';
import { setNewUserId, getNewUserId } from './useRandomId';

// 创建实例时配置默认值
const service = axios.create({
    baseURL: '/',
    timeout: 30000,
    responseType: 'json'
});

// 请求拦截器
service.interceptors.request.use(config => {
    if (config.url.includes('stream')) {
        config.timeout = 3000;
    }
    if (window.location.search) {
        config.url += window.location.search;
    }
    Object.assign(config.headers, ajaxHeader());
    return config;
});

// 响应拦截器
service.interceptors.response.use(
    response => {
        let res = response.data;
        if (response?.status === 200) {
            return Promise.resolve({
                code: 0,
                message: '',
                data: res
            });
        }
        return Promise.resolve({ code: -1, message: '网络异常，请稍后再试', data: null });
    },
    error => {
        const res = { code: -1, message: error?.response?.data?.detail || '网络异常，请稍后再试', data: null };
        return Promise.resolve(res);
    }
);

export const ajaxHeader = () => {
    if (!localStorage.getItem('uid')) {
        setNewUserId();
    }
    return {
        'Content-Type': 'application/json;charset=UTF-8',
        Accept: 'application/json',
        service: 'minicpmo-server',
        uid: getNewUserId()
    };
};

export default {
    get(url, params, config = {}) {
        return service.get(url, { params, ...config });
    },
    post(url, data, config = {}) {
        return service.post(url, data, { ...config });
    }
};

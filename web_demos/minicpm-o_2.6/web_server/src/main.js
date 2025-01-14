import './styles/main.css';

import { router, setupRouter } from '@/router';
import { setupRouterGuard } from '@/router/guard';
import SvgIcon from '@/components/SvgIcon/index.vue';
import { createI18n } from 'vue-i18n';

import App from './App.vue';
import en from './i18n/en.json';
import zh from './i18n/zh.json';

const savedLanguage = localStorage.getItem('language') || 'zh';

const i18n = createI18n({
    locale: savedLanguage, // 默认语言
    messages: {
        en,
        zh
    }
});

const app = createApp(App);

// Configure routing
// 配置路由
setupRouter(app);

// router-guard
// 路由守卫
setupRouterGuard(router);

// Register global directive
// 注册全局指令
// setupGlobDirectives(app);

app.component('SvgIcon', SvgIcon);

app.use(i18n);

app.mount('#app');

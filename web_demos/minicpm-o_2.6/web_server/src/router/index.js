import { createRouter, createWebHistory } from 'vue-router';
import { basicRoutes } from './menu';

// 创建一个可以被 Vue 应用程序使用的路由实例
export const router = createRouter({
    // 创建一个 hash 历史记录。
    history: createWebHistory(import.meta.env.BASE_URL),
    // 路由列表。
    routes: basicRoutes
});

// config router
// 配置路由器
export function setupRouter(app) {
    app.use(router);
}

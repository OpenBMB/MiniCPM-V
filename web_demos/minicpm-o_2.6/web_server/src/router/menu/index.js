export const basicRoutes = [
    {
        path: '/',
        component: () => import('@/views/home/index.vue')
    },
    {
        path: '/:port',
        component: () => import('@/views/home/index.vue')
    }
];

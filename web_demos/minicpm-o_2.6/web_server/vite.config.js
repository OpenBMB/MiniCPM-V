import { fileURLToPath, URL } from 'node:url';

import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
// import vueDevTools from 'vite-plugin-vue-devtools';

import Icons from 'unplugin-icons/vite';
import IconsResolver from 'unplugin-icons/resolver';
import AutoImport from 'unplugin-auto-import/vite';
import Components from 'unplugin-vue-components/vite';
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers';
import fs from 'fs';
import path from 'path';

export default defineConfig({
    plugins: [
        vue(),
        // vueDevTools(),
        AutoImport({
            resolvers: [
                ElementPlusResolver(), // Auto import icon components
                // 自动导入图标组件
                IconsResolver({
                    prefix: 'Icon'
                })
            ],
            imports: ['vue', 'vue-router', '@vueuse/core'],
            dirs: ['src/apis/**/*', 'src/hooks/*'],
            vueTemplate: true,
            eslintrc: {
                enabled: true
            }
        }),
        Components({
            resolvers: [
                ElementPlusResolver(), // 自动注册图标组件
                IconsResolver({
                    enabledCollections: ['ep']
                })
            ],
            dirs: ['src/components']
        }),
        Icons({
            autoInstall: true
        })
    ],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url))
        }
    },
    css: {
        preprocessorOptions: {
            less: {
                additionalData: `@import 'src/styles/element/index.less';`
            }
        }
    },
    server: {
        https: {
            key: fs.readFileSync(path.resolve(__dirname, 'key.pem')),
            cert: fs.readFileSync(path.resolve(__dirname, 'cert.pem')),
        },
        host: '0.0.0.0',
        port: 8088,
        proxy: {
            '/api/v1': {
                target: 'http://127.0.0.1:32550',
                ws: true,
                changeOrigin: true
            },
            '/ws': {
                target: 'http://127.0.0.1:32550',
                ws: true,
                changeOrigin: true
            }
        }
    }
});

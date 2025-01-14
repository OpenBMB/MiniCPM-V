/* eslint-env node */
require('@rushstack/eslint-patch/modern-module-resolution');

module.exports = {
    root: true,
    extends: [
        'plugin:vue/vue3-essential',
        'eslint:recommended',
        '@vue/eslint-config-prettier/skip-formatting',
        './.eslintrc-auto-import.json',
    ],
    parserOptions: {
        ecmaVersion: 'latest',
    },
    rules: {
        'no-console': process.env.NODE_ENV === 'production' ? 'off' : 'warn',
        'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
        'no-var': process.env.NODE_ENV === 'production' ? 'off' : 'warn',
        'no-undef': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
        'vue/multi-word-component-names': 'off', // 不校验组件名
        'no-empty': 0, // 允许代码块为空
        'vue/no-unused-components': 'warn',
        'no-unused-vars': 'warn',
        'prettier/prettier': 'off', // 不符合prettier格式规范的编码eslint直接自动报错
    },
};

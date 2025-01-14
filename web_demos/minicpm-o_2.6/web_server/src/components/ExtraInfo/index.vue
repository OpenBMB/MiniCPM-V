<template>
    <div class="extra-info">
        <div class="model-version" v-if="modelVersion">模型版本: {{ modelVersion }}</div>
        <div class="web-version">前端版本: {{ webVersion }}</div>
    </div>
</template>

<script setup>
    defineProps({
        modelVersion: {
            type: String,
            default: ''
        },
        webVersion: {
            type: String,
            default: ''
        }
    });
</script>

<style lang="less" scoped>
    .extra-info {
        position: fixed;
        top: 62px;
        left: 4vw;
        display: flex;
        .model-version,
        .web-version {
            font-size: 12px;
            color: red;
        }
        .model-version {
            margin-right: 16px;
        }
    }
</style>

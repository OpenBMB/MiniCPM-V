<template>
    <div class="delay-tips">
        <span>当前发生延迟，目前延迟{{ delayTimestamp }}ms，积压{{ delayCount * 200 }}ms未发</span>
    </div>
</template>
<script setup>
    defineProps({
        delayTimestamp: {
            type: Number,
            defalult: 0
        },
        delayCount: {
            type: Number,
            defalult: 0
        }
    });
</script>
<style lang="less" scoped>
    .delay-tips {
        font-size: 12px;
        color: #dc3545;
    }
</style>

<template>
    <div class="ideas">
        <div class="ideas-title">
            <img src="@/assets/images/ideas-icon.png " />
            <span>Convsersation ideas</span>
        </div>
        <div class="ideas-content">
            <div class="ideas-content-item" v-for="(item, index) in ideasList" :key="index">{{ item }}</div>
        </div>
    </div>
</template>

<script setup>
    defineProps({
        ideasList: {
            type: Array,
            default: () => []
        }
    });
</script>

<style lang="less" scoped>
    .ideas {
        margin-top: 16px;
        box-shadow: 0 0 0 0.5px #e0e0e0;
        border-radius: 12px;
        padding: 18px 28px;
        &-title {
            font-size: 20px;
            font-weight: 500;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            img {
                width: 24px;
                height: 24px;
                margin-right: 10px;
            }
            span {
                color: #171717;
                font-family: PingFang SC;
                font-size: 16px;
                font-style: normal;
                font-weight: 500;
                line-height: normal;
            }
        }
        &-content {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            &-item {
                display: flex;
                align-items: center;
                border-radius: 10px;
                background: #eaefff;
                padding: 10px 24px;
                color: #7579eb;
                font-family: PingFang SC;
                font-size: 14px;
                font-style: normal;
                font-weight: 400;
                line-height: normal;
            }
        }
    }
</style>

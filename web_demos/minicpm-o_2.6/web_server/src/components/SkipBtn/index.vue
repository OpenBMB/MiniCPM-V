<template>
    <div :class="`skip-btn ${disabled ? 'disabled-btn' : ''}`">
        <div class="pause-icon">
            <SvgIcon name="pause" className="pause-svg" />
        </div>
        <span class="btn-text">{{ t('skipMessageBtn') }}</span>
    </div>
</template>
<script setup>
    import { useI18n } from 'vue-i18n';

    const { t } = useI18n();
    defineProps({
        disabled: {
            type: Boolean,
            default: false
        }
    });
</script>
<style lang="less">
    .skip-btn {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        padding: 8px 14px 8px 10px;
        border-radius: 90px;
        background: #5865f2;
        cursor: pointer;
        user-select: none;
        .pause-icon {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 32px;
            height: 32px;
            background: #ffffff;
            border-radius: 50%;
            margin-right: 8px;
            .pause-svg {
                width: 18px;
                height: 18px;
                color: #5865f2;
            }
        }
        .btn-text {
            color: #fff;
            font-family: PingFang SC;
            font-size: 16px;
            font-style: normal;
            font-weight: 400;
            line-height: normal;
        }
    }
    .disabled-btn {
        cursor: not-allowed;
        background: #f3f3f3;
        .pause-icon {
            background: #d1d1d1;
            .pause-svg {
                color: #ffffff;
            }
        }
        .btn-text {
            color: #d1d1d1;
        }
    }
</style>

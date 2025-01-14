<template>
    <div class="time">
        <div class="time-minute">{{ minute || '00' }}</div>
        <div class="time-colon">:</div>
        <div class="time-second">{{ second || '00' }}</div>
    </div>
</template>

<script setup>
    import { limitTime, tipsRemainingTime } from '@/enums';

    const start = defineModel();

    const emits = defineEmits(['timeUp']);

    const remainingTime = ref();
    const minute = ref();
    const second = ref();
    const timeInterval = ref(null);

    const startCount = () => {
        remainingTime.value = limitTime;
        updateCountDown();
        timeInterval.value = setInterval(() => {
            updateCountDown();
        }, 1000);
    };
    const updateCountDown = () => {
        let minutes = Math.floor(remainingTime.value / 60);
        let seconds = remainingTime.value % 60;

        // 格式化分钟和秒，确保它们是两位数
        minute.value = minutes < 10 ? '0' + minutes : minutes;
        second.value = seconds < 10 ? '0' + seconds : seconds;

        // 剩余1分钟提示用户
        if (remainingTime.value === tipsRemainingTime) {
            ElMessage({
                type: 'warning',
                message: `This call will disconnect in ${tipsRemainingTime} seconds.`,
                duration: 3000,
                customClass: 'time-warning'
            });
        }
        // 防止倒计时变成负数
        if (remainingTime.value > 0) {
            remainingTime.value--;
        } else {
            clearInterval(timeInterval);
            emits('timeUp');
        }
    };
    watch(
        () => start.value,
        newVal => {
            timeInterval.value && clearInterval(timeInterval.value);
            if (newVal) {
                startCount();
            }
        },
        { immediate: true }
    );
</script>
<style lang="less" scoped>
    .time {
        display: flex;
        align-items: center;
        .time-minute,
        .time-second {
            width: 26px;
            height: 26px;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 3.848px;
            background: rgba(47, 47, 47, 0.5);
        }
        .time-colon {
            margin: 0 3px;
        }
    }
</style>

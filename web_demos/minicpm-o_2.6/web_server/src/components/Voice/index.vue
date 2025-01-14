<template>
    <div class="bars" id="bars" :style="boxStyle">
        <!-- 柱形条 -->
        <div class="bar" v-for="(item, index) in defaultList" :key="index" :style="itemAttr(item)"></div>
    </div>
</template>

<script setup>
    const props = defineProps({
        analyser: {
            type: Object
        },
        dataArray: {
            type: [Array, Uint8Array]
        },
        isCalling: {
            type: Boolean,
            default: false
        },
        isPlaying: {
            type: Boolean,
            default: false
        },
        // 容器高度
        boxStyle: {
            type: Object,
            default: () => {
                return {
                    height: '80px'
                };
            }
        },
        // 柱形条宽度
        itemStyle: {
            type: Object,
            default: () => {
                return {
                    width: '6px',
                    margin: '0 2px',
                    borderRadius: '5px'
                };
            }
        },
        configList: {
            type: Array,
            default: () => []
        }
    });
    const animationFrameId = ref();
    const defaultList = ref([]);
    const bgColor = ref('#4c5cf8');
    const itemAttr = computed(() => item => {
        return {
            height: item + 'px',
            ...props.itemStyle
        };
    });
    watch(
        () => props.dataArray,
        newVal => {
            if (newVal && props.isCalling) {
                console.log('draw');
                drawBars();
            } else {
                console.log('stop');
                stopDraw();
            }
        }
    );
    watch(
        () => props.configList,
        newVal => {
            if (newVal.length > 0) {
                defaultList.value = newVal;
            }
        },
        { immediate: true }
    );
    watch(
        () => props.isPlaying,
        newVal => {
            if (newVal) {
                // 绿色
                bgColor.value = '#4dc100';
            } else {
                // 蓝色
                bgColor.value = '#4c5cf8';
            }
        }
    );
    function drawBars() {
        const bars = document.querySelectorAll('.bar');
        if (bars.length === 0) {
            cancelAnimationFrame(animationFrameId.value);
            return;
        }

        const maxHeight = document.querySelector('.bars').clientHeight; // 最大高度为容器的高度

        const averageVolume = props.dataArray.reduce((sum, value) => sum + value, 0) / props.dataArray.length;
        const normalizedVolume = props.isPlaying ? Math.random() : averageVolume / 128; // 将音量数据归一化为0到1之间

        bars.forEach((bar, index) => {
            const minHeight = defaultList.value[index];
            const randomFactor = Math.random() * 1.5 + 0.5; // 随机因子
            const newHeight = Math.min(
                maxHeight,
                minHeight + (maxHeight - minHeight) * normalizedVolume * randomFactor
            ); // 根据音量设置高度
            bar.style.height = `${newHeight}px`; // 设置新的高度
            bar.style.backgroundColor = bgColor.value;
        });

        animationFrameId.value = requestAnimationFrame(drawBars);
    }
    const stopDraw = () => {
        if (animationFrameId.value) {
            cancelAnimationFrame(animationFrameId.value);
        }
    };
</script>

<style lang="less" scoped>
    .bars {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .bar {
        // width: 6px;
        // margin: 0 2px;
        background-color: #4c5cf8;
        transition:
            height 0.1s,
            background-color 0.1s;
        border-radius: 5px; /* 圆角 */
    }
</style>

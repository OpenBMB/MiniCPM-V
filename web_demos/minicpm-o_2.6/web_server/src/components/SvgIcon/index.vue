<template>
    <svg :class="iconClass" v-html="content"></svg>
</template>

<script setup>
    const props = defineProps({
        name: {
            type: String,
            required: true
        },
        className: {
            type: String,
            default: ''
        }
    });

    const content = ref('');

    const iconClass = computed(() => ['svg-icon', props.className]);
    onMounted(() => {
        import(`@/assets/svg/${props.name}.svg`)
            .then(module => {
                fetch(module.default)
                    .then(response => response.text())
                    .then(svg => {
                        content.value = svg;
                    });
            })
            .catch(error => {
                console.error(`Error loading SVG icon: ${props.name}`, error);
            });
    });
</script>
<style lang="less" scoped>
    .svg-icon {
        width: 24px;
        height: 24px;
    }
</style>

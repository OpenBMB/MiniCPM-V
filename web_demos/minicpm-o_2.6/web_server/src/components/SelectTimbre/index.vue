<template>
    <div class="select-timbre">
        <el-checkbox-group v-model="timbre" @change="handleSelectTimbre" :disabled="disabled">
            <el-checkbox :value="1" label="Default Audio"></el-checkbox>
            <!-- <el-upload
                v-model:file-list="fileList"
                action=""
                :multiple="false"
                :on-change="handleChangeFile"
                :auto-upload="false"
                :show-file-list="false"
                :disabled="disabled"
                accept="audio/*"
            >
                <el-checkbox :value="2">
                    <span>Customization: Upload Audio</span>
                    <SvgIcon name="upload" className="checkbox-icon" />
                </el-checkbox>
            </el-upload> -->
        </el-checkbox-group>
    </div>
</template>

<script setup>
    const timbre = defineModel('timbre');
    const audioData = defineModel('audioData');
    const disabled = defineModel('disabled');
    const fileList = ref([]);

    const handleSelectTimbre = e => {
        if (e.length > 1) {
            const val = e[e.length - 1];
            timbre.value = [val];
            // 默认音色
            if (val === 1) {
                audioData.value = {
                    base64Str: '',
                    type: 'mp3'
                };
            }
        }
    };
    const handleChangeFile = file => {
        if (isAudio(file) && sizeNotExceed(file)) {
            fileList.value = [file];
            timbre.value = [2];
            handleUpload();
        } else {
            ElMessage.error('Please upload audio file and size not exceed 1MB');
        }
    };
    const isAudio = file => {
        return file.name.endsWith('.mp3') || file.name.endsWith('.wav');
    };
    const sizeNotExceed = file => {
        return file.size / 1024 / 1024 <= 1;
    };
    const handleUpload = async () => {
        const file = fileList.value[0].raw;
        if (file) {
            const reader = new FileReader();
            reader.onload = e => {
                const base64String = e.target.result.split(',')[1];
                audioData.value = {
                    base64Str: base64String,
                    type: file.name.split('.')[1]
                };
            };
            reader.readAsDataURL(file);
        }
    };
</script>
<style lang="less">
    .select-timbre {
        display: flex;
        align-items: center;
        .el-checkbox-group {
            display: flex;
            > .el-checkbox {
                margin-right: 12px;
            }
        }
        .el-checkbox {
            padding: 8px 16px;
            border-radius: 10px;
            background: #eaefff;
            margin-right: 0;
            height: 40px;
            .el-checkbox__input {
                .el-checkbox__inner {
                    border: 1px solid #4dc100;
                }
            }
            .el-checkbox__input.is-checked {
                .el-checkbox__inner {
                    background: #4dc100;
                }
            }
            .el-checkbox__input.is-checked.is-disabled {
                .el-checkbox__inner::after {
                    border-color: #ffffff;
                }
            }
        }
        .el-checkbox__label {
            color: #7579eb !important;
            font-family: PingFang SC;
            font-size: 16px;
            font-style: normal;
            font-weight: 400;
            line-height: normal;
            display: flex;
            align-items: center;
            .checkbox-icon {
                margin-left: 4px;
            }
        }
        .el-checkbox + .el-checkbox {
            margin-left: 12px;
        }
    }
</style>

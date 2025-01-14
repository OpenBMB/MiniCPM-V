<template>
    <div class="user-config">
        <div class="user-config-title">模型配置</div>
        <div class="config-item">
            <div class="config-item-label">语音打断：</div>
            <div class="config-item-content">
                <el-switch
                    v-model="configData.canStopByVoice"
                    inline-prompt
                    active-text="是"
                    inactive-text="否"
                    size="small"
                    :disabled="isCalling"
                />
            </div>
        </div>
        <div class="config-item">
            <div class="config-item-label">视频画质：</div>
            <div class="config-item-content">
                <el-radio-group v-model="configData.videoQuality" :disabled="isCalling">
                    <el-radio :value="true">高清</el-radio>
                    <el-radio :value="false">低清</el-radio>
                </el-radio-group>
            </div>
        </div>
        <div class="config-item">
            <div class="config-item-label">VAD阈值：</div>
            <div class="config-item-content vad-slider">
                <el-slider
                    v-model="configData.vadThreshold"
                    :min="0.5"
                    :max="1"
                    :step="0.1"
                    size="small"
                    :disabled="isCalling"
                />
            </div>
        </div>
        <!-- <div class="timbre-model">
            <div class="timbre-model-label">音色人物：</div>
            <div class="timbre-model-content">
                <el-select
                    v-model="configData.timbreId"
                    style="width: 100%"
                    @change="handleChangePeople"
                    clearable
                    placeholder="请选择"
                >
                    <el-option v-for="item in peopleList" :key="item.id" :value="item.id" :label="item.name">
                        {{ item.name }}
                    </el-option>
                </el-select>
            </div>
        </div> -->
        <div class="prompt-item">
            <div class="prompt-item-label">Assistant_prompt：</div>
            <div class="prompt-item-content">
                <el-input
                    type="textarea"
                    :rows="3"
                    v-model="configData.assistantPrompt"
                    resize="none"
                    :disabled="isCalling"
                />
            </div>
        </div>
        <div class="config-item">
            <div class="config-item-label">使用语音prompt：</div>
            <div class="config-item-content">
                <el-switch
                    v-model="configData.useAudioPrompt"
                    inline-prompt
                    active-text="是"
                    inactive-text="否"
                    size="small"
                    :disabled="isCalling"
                    @change="handleSelectUseAudioPrompt"
                />
            </div>
        </div>
        <div class="voice-prompt-box">
            <div class="prompt-item" v-if="configData.useAudioPrompt">
                <div class="prompt-item-label">Voice_clone_prompt：</div>
                <div class="prompt-item-content">
                    <el-input
                        type="textarea"
                        :rows="8"
                        v-model="configData.voiceClonePrompt"
                        resize="none"
                        :disabled="isCalling"
                    />
                </div>
            </div>

            <div class="timbre-config" v-if="configData.useAudioPrompt">
                <div class="timbre-config-label">音色选择：</div>
                <div class="timbre-config-content">
                    <el-checkbox-group v-model="configData.timbre" @change="handleSelectTimbre" :disabled="isCalling">
                        <el-checkbox :value="1" label="Default Audio"></el-checkbox>
                        <el-upload
                            v-model:file-list="fileList"
                            action=""
                            :multiple="false"
                            :on-change="handleChangeFile"
                            :auto-upload="false"
                            :show-file-list="false"
                            :disabled="isCalling"
                            accept="audio/*"
                        >
                            <el-checkbox :value="2">
                                <!-- <span>Customization: Upload Audio</span> -->
                                <span>Customization</span>
                                <SvgIcon name="upload" className="checkbox-icon" />
                            </el-checkbox>
                        </el-upload>
                    </el-checkbox-group>
                </div>
            </div>
            <div class="file-content" v-if="fileName">
                <SvgIcon name="document" class="document-icon" />
                <span class="file-name">{{ fileName }}</span>
            </div>
        </div>
    </div>
</template>

<script setup>
    const isCalling = defineModel('isCalling');
    const type = defineModel('type');

    let defaultVoiceClonePrompt =
        '你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。模仿输入音频中的声音特征。';
    let defaultAssistantPrompt = '作为助手，你将使用这种声音风格说话。';

    const fileList = ref([]);
    const fileName = ref('');

    const configData = ref({
        canStopByVoice: false,
        videoQuality: false,
        useAudioPrompt: true,
        vadThreshold: 0.8,
        voiceClonePrompt: defaultVoiceClonePrompt,
        assistantPrompt: defaultAssistantPrompt,
        timbre: [1],
        audioFormat: 'mp3',
        base64Str: '',
        timbreId: ''
    });

    const peopleList = [
        {
            id: 1,
            name: 'Trump',
            voiceClonePrompt: '',
            assistantPrompt: ''
        },
        {
            id: 2,
            name: '说相声',
            voiceClonePrompt: '克隆音频提示中的音色以生成语音',
            assistantPrompt: '请角色扮演这段音频，请以相声演员的口吻说话'
        },
        {
            id: 3,
            name: '默认',
            voiceClonePrompt: defaultVoiceClonePrompt,
            assistantPrompt: defaultAssistantPrompt
        }
    ];
    watch(
        () => type.value,
        val => {
            if (val === 'video') {
                console.log('val: ', val);
                defaultVoiceClonePrompt =
                    '你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。模仿输入音频中的声音特征。';
                defaultAssistantPrompt = '作为助手，你将使用这种声音风格说话。';
            } else {
                defaultVoiceClonePrompt = '克隆音频提示中的音色以生成语音。';
                defaultAssistantPrompt = 'Your task is to be a helpful assistant using this voice pattern.';
            }
            configData.value.voiceClonePrompt = defaultVoiceClonePrompt;
            configData.value.assistantPrompt = defaultAssistantPrompt;
        },
        { immediate: true }
    );
    onMounted(() => {
        handleSetStorage();
    });
    const handleSelectTimbre = e => {
        if (e.length > 1) {
            const val = e[e.length - 1];
            configData.value.timbre = [val];
            // 默认音色
            if (val === 1) {
                configData.value.audioFormat = 'mp3';
                configData.value.base64Str = '';
                fileList.value = [];
                fileName.value = '';
            }
        }
    };
    const handleChangeFile = file => {
        if (isAudio(file) && sizeNotExceed(file)) {
            fileList.value = [file];
            fileName.value = file.name;
            configData.value.timbre = [2];
            handleUpload();
        } else {
            ElMessage.error('Please upload audio file and size not exceed 10MB');
        }
    };
    const isAudio = file => {
        return file.raw.type.includes('audio');
    };
    const sizeNotExceed = file => {
        return file.size / 1024 / 1024 <= 10;
    };
    const handleUpload = async () => {
        const file = fileList.value[0].raw;
        if (file) {
            const reader = new FileReader();
            reader.onload = e => {
                const base64String = e.target.result.split(',')[1];
                configData.value.audioFormat = file.name.split('.')[1];
                configData.value.base64Str = base64String;
            };
            reader.readAsDataURL(file);
        }
    };
    const handleSelectUseAudioPrompt = val => {
        if (val) {
            configData.value.voiceClonePrompt = defaultVoiceClonePrompt;
            configData.value.assistantPrompt = defaultAssistantPrompt;
        }
    };
    // 配置发生变化，更新到localstorage中
    watch(configData.value, () => {
        handleSetStorage();
    });
    const handleSetStorage = () => {
        const { timbre, canStopByVoice, ...others } = configData.value;
        const defaultConfigData = {
            canStopByVoice,
            ...others
        };
        localStorage.setItem('configData', JSON.stringify(defaultConfigData));
        localStorage.setItem('canStopByVoice', canStopByVoice);
    };
    const handleChangePeople = val => {
        console.log('val: ', val);
        if (!val) {
            return;
        }
        const index = peopleList.findIndex(item => item.id === val);
        configData.value.voiceClonePrompt = peopleList[index].voiceClonePrompt;
        configData.value.assistantPrompt = peopleList[index].assistantPrompt;
        configData.value.timbre = [1];
    };
</script>
<style lang="less">
    .user-config {
        &-title {
            height: 61px;
            padding: 18px 18px 0;
            color: rgba(23, 23, 23, 0.9);
            font-family: PingFang SC;
            font-size: 16px;
            font-style: normal;
            font-weight: 500;
            line-height: normal;
        }
        .config-item {
            display: flex;
            align-items: center;
            width: 100%;
            padding: 0 0 0 18px;
            margin-bottom: 12px;
            &-label {
                width: 120px;
                flex-shrink: 0;
            }
            &-content {
                flex: 1;
                margin-left: 16px;
                .el-radio-group {
                    .el-radio {
                        width: 50px;
                    }
                }
            }
            &-content.vad-slider {
                width: 80%;
                padding-left: 7px;
                margin-right: 20px;
                .el-slider__button {
                    width: 14px;
                    height: 14px;
                }
            }
        }
        .timbre-config {
            padding: 0 0 0 18px;
            &-label {
                margin-bottom: 12px;
            }
            &-content {
                display: flex;
                align-items: center;
                .el-checkbox-group {
                    display: flex;
                    flex-wrap: wrap;
                    flex: 1;
                    > .el-checkbox {
                        margin-right: 12px;
                    }
                }
                .el-checkbox {
                    padding: 8px 16px;
                    border-radius: 10px;
                    background: #eaefff;
                    margin-bottom: 12px;
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
        }
        .prompt-item {
            // padding: 0 0 0 18px;
            margin-bottom: 12px;
            &-label {
                // margin-bottom: 16px;
            }
        }
        .file-content {
            padding: 0 0 0 18px;
            font-size: 14px;
            display: flex;
            align-items: center;
            .document-icon {
                width: 16px;
                height: 16px;
                margin-right: 4px;
            }
            .file-name {
                flex: 1;
                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;
            }
        }
        .timbre-model {
            padding: 0 0 0 18px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            &-label {
                width: 120px;
                flex-shrink: 0;
            }
            &-content {
                flex: 1;
                margin-left: 16px;
            }
        }
        .voice-prompt-box {
            border: 1px solid #eaefff;
            margin-left: 18px;
            padding: 12px;
            width: 50%;
        }
    }
</style>

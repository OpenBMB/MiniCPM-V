<template>
    <div :class="`user-config ${t('modelConfigTitle') === '模型配置' ? '' : 'en-user-config'}`">
        <div class="user-config-title">{{ t('modelConfigTitle') }}</div>
        <div class="config-item">
            <div class="config-item-label">
                <span>{{ t('audioInterruptionBtn') }}</span>
                <el-tooltip class="box-item" effect="dark" :content="t('audioInterruptionTips')" placement="top">
                    <SvgIcon name="question" class="question-icon" /> </el-tooltip
                >:
            </div>
            <div class="config-item-content">
                <el-switch
                    v-model="configData.canStopByVoice"
                    inline-prompt
                    :active-text="t('yes')"
                    :inactive-text="t('no')"
                    size="small"
                    :disabled="isCalling"
                />
            </div>
        </div>
        <div class="config-item" v-if="type === 'video'">
            <div class="config-item-label">
                <span>{{ t('videoQualityBtn') }}</span>
                <el-tooltip class="box-item" effect="dark" :content="t('videoQualityTips')" placement="top">
                    <SvgIcon name="question" class="question-icon" /> </el-tooltip
                >:
            </div>
            <div class="config-item-content">
                <el-switch
                    v-model="configData.videoQuality"
                    inline-prompt
                    :active-text="t('yes')"
                    :inactive-text="t('no')"
                    size="small"
                    :disabled="isCalling"
                />
            </div>
        </div>
        <div class="config-item">
            <div class="config-item-label">
                <span>{{ t('vadThresholdBtn') }}</span>
                <el-tooltip class="box-item" effect="dark" :content="t('vadThresholdTips')" placement="top">
                    <SvgIcon name="question" class="question-icon" /> </el-tooltip
                >:
            </div>
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
        <div class="prompt-item" v-if="type === 'voice'">
            <div class="prompt-item-label">
                <span>{{ t('assistantPromptBtn') }}</span>
                <el-tooltip class="box-item" effect="dark" :content="t('assistantPromptTips')" placement="top">
                    <SvgIcon name="question" class="question-icon" /> </el-tooltip
                >:
            </div>
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
        <!-- <div class="config-item">
            <div class="config-item-label">{{ t('useVoicePromptBtn') }}:</div>
            <div class="config-item-content">
                <el-switch
                    v-model="configData.useAudioPrompt"
                    inline-prompt
                    :active-text="t('yes')"
                    :inactive-text="t('no')"
                    size="small"
                    :disabled="isCalling"
                    @change="handleSelectUseAudioPrompt"
                />
            </div>
        </div> -->
        <div class="timbre-model">
            <div class="timbre-model-label">
                <span>{{ t('toneColorOptions') }}</span>
                <el-tooltip class="box-item" effect="dark" :content="t('toneColorOptionsTips')" placement="top">
                    <SvgIcon name="question" class="question-icon" /> </el-tooltip
                >:
            </div>
            <div class="timbre-model-content">
                <el-select
                    v-model="configData.useAudioPrompt"
                    style="width: 100%"
                    @change="handleChangePeople"
                    placeholder="请选择"
                    :disabled="isCalling"
                >
                    <el-option :value="0" :label="t('nullOption')">{{ t('nullOption') }}</el-option>
                    <el-option :value="1" :label="t('defaultOption')">{{ t('defaultOption') }}</el-option>
                    <el-option :value="2" :label="t('femaleOption')">{{ t('femaleOption') }}</el-option>
                    <el-option :value="3" :label="t('maleOption')">{{ t('maleOption') }}</el-option>
                </el-select>
            </div>
        </div>
        <!-- <div class="prompt-item">
            <div class="prompt-item-label">
                <span>{{ t('voiceClonePromptInput') }}</span>
                <el-tooltip class="box-item" effect="dark" :content="t('voiceClonePromptTips')" placement="top">
                    <SvgIcon name="question" class="question-icon" /> </el-tooltip
                >:
            </div>
            <div class="prompt-item-content">
                <el-input
                    type="textarea"
                    :rows="3"
                    v-model="configData.voiceClonePrompt"
                    resize="none"
                    :disabled="true"
                />
            </div>
        </div> -->
        <!-- <div class="timbre-config" v-if="configData.useAudioPrompt">
            <div class="timbre-config-label">{{ t('audioChoiceBtn') }}:</div>
            <div class="timbre-config-content">
                <el-checkbox-group v-model="configData.timbre" @change="handleSelectTimbre" :disabled="isCalling">
                    <el-checkbox :value="1" :label="t('defaultAudioBtn')"></el-checkbox>
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
                            <span>{{ t('customizationBtn') }}</span>
                            <SvgIcon name="upload" className="checkbox-icon" />
                        </el-checkbox>
                    </el-upload>
                </el-checkbox-group>
            </div>
        </div>
        <div class="file-content" v-if="fileName">
            <SvgIcon name="document" class="document-icon" />
            <span class="file-name">{{ fileName }}</span>
        </div> -->
    </div>
</template>

<script setup>
    const isCalling = defineModel('isCalling');
    const type = defineModel('type');
    import { useI18n } from 'vue-i18n';

    const { t, locale } = useI18n();

    let defaultVoiceClonePrompt =
        '你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。模仿输入音频中的声音特征。';
    let defaultAssistantPrompt = '';

    const fileList = ref([]);
    const fileName = ref('');

    const configData = ref({
        canStopByVoice: false,
        videoQuality: false,
        useAudioPrompt: 1,
        vadThreshold: 0.8,
        voiceClonePrompt: defaultVoiceClonePrompt,
        assistantPrompt: defaultAssistantPrompt,
        timbre: [1],
        audioFormat: 'mp3',
        base64Str: ''
    });

    // let peopleList = [];
    // watch(
    //     () => type.value,
    //     val => {
    //         console.log('val: ', val);
    //         if (val === 'video') {
    //             defaultVoiceClonePrompt =
    //                 '你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。模仿输入音频中的声音特征。';
    //             defaultAssistantPrompt = '作为助手，你将使用这种声音风格说话。';
    //         } else {
    //             defaultVoiceClonePrompt = '克隆音频提示中的音色以生成语音。';
    //             defaultAssistantPrompt = 'Your task is to be a helpful assistant using this voice pattern.';
    //         }
    //         configData.value.voiceClonePrompt = defaultVoiceClonePrompt;
    //         configData.value.assistantPrompt = defaultAssistantPrompt;
    //     },
    //     { immediate: true }
    // );
    watch(
        locale,
        (newLocale, oldLocale) => {
            console.log(`Language switched from ${oldLocale} to ${newLocale}`);
            if (newLocale === 'zh' && type.value === 'video') {
                defaultAssistantPrompt = '作为助手，你将使用这种声音风格说话。';
            } else if (newLocale === 'zh' && type.value === 'voice') {
                defaultAssistantPrompt = '作为助手，你将使用这种声音风格说话。';
            } else if (newLocale === 'en' && type.value === 'video') {
                defaultAssistantPrompt = 'As an assistant, you will speak using this voice style.';
            } else {
                defaultAssistantPrompt = 'As an assistant, you will speak using this voice style.';
            }
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
        // const index = peopleList.findIndex(item => item.id === val);
        configData.value.voiceClonePrompt = defaultVoiceClonePrompt;
        configData.value.assistantPrompt = defaultAssistantPrompt;
        configData.value.timbre = [1];
    };
</script>
<style lang="less" scoped>
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
            margin-bottom: 20px;
            &-label {
                width: 120px;
                flex-shrink: 0;
                display: flex;
                align-items: center;
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
                margin-bottom: 20px;
                display: flex;
                align-items: center;
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
            padding: 0 0 0 18px;
            margin-bottom: 20px;
            &-label {
                // margin-bottom: 16px;
                display: flex;
                align-items: center;
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
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            &-label {
                width: 120px;
                flex-shrink: 0;
                display: flex;
                align-items: center;
            }
            &-content {
                flex: 1;
                margin-left: 16px;
            }
        }
    }
    .en-user-config {
        .config-item-label {
            width: 160px;
        }
        .timbre-model-label {
            width: 160px;
        }
    }
    .question-icon {
        width: 14px;
        height: 14px;
        cursor: pointer;
        margin-left: 6px;
    }
</style>
<style lang="less">
    .el-switch--small .el-switch__core {
        min-width: 50px;
    }
    .el-popper.is-dark {
        max-width: 300px;
    }
</style>

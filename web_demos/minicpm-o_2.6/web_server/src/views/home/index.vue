<template>
    <div class="home-page">
        <div class="home-page-header">
            <div class="home-page-header-logo">
                <!-- <img src="@/assets/images/logo.png" /> -->
                <SvgIcon name="miniCPM2.6" class="logo-icon" />
            </div>
            <div class="home-page-header-menu">
                <div
                    class="home-page-header-menu-item"
                    v-for="(item, index) in tabList"
                    :key="item.type"
                    :class="`home-page-header-menu-item ${activeTab === item.type ? 'active-tab' : ''} ${item.disabled ? 'disabled-tab' : ''}`"
                    @click="handleClickTab(item.type, index)"
                >
                    {{ getMenuTab(item.type) }}
                </div>
            </div>

            <div class="home-page-header-switch">
                <div class="change-language">
                    <div
                        :class="`change-language-item ${language === 'en' ? 'active-language' : ''}`"
                        @click="handleChangeLanguage('en')"
                    >
                        English
                    </div>
                    <div
                        :class="`change-language-item ${language === 'zh' ? 'active-language' : ''}`"
                        @click="handleChangeLanguage('zh')"
                    >
                        中文
                    </div>
                </div>
            </div>
        </div>
        <div :class="`home-page-content ${activeTab === 'chatbot' && 'no-padding'}`">
            <VoiceCallWs v-if="isWebSocket && activeTab === 'voice'" v-model="isCalling" />
            <VoiceCall v-else-if="!isWebSocket && activeTab === 'voice'" v-model="isCalling" />
            <VideoCallWs v-else-if="isWebSocket && activeTab === 'video'" v-model="isCalling" />
            <VideoCall v-else-if="!isWebSocket && activeTab === 'video'" v-model="isCalling" />
            <!-- TODO: https is required to support chatbot in iframe -->
            <iframe
                src="http://127.0.0.1:8000/"
                frameborder="0"
                width="100%"
                height="100%"
                v-else
            />
            <div class="config-box" v-if="activeTab !== 'chatbot'">
                <ModelConfig v-model:isCalling="isCalling" v-model:type="activeTab" />
            </div>
        </div>
    </div>
</template>

<script setup>
    import VoiceCall from './components/VoiceCall.vue';
    import VoiceCallWs from './components/VoiceCall_0105.vue';
    import VideoCall from './components/VideoCall.vue';
    import VideoCallWs from './components/VideoCall_0105.vue';
    import { useI18n } from 'vue-i18n';
    import { useRoute, useRouter } from 'vue-router';

    const route = useRoute();
    const router = useRouter();
    const typeObj = {
        0: 'video',
        1: 'voice',
        2: 'chatbot'
    };
    const defaultType = typeObj[route.query.type] || 'voice';

    const { t, locale } = useI18n();
    const activeTab = ref(defaultType);
    const language = ref(localStorage.getItem('language') || 'zh');
    const isWebSocket = false;
    const tabList = ref([
        {
            type: 'video',
            text: 'Realtime Video Call'
        },
        {
            type: 'voice',
            text: 'Realtime Voice Call'
        },
        {
            type: 'chatbot',
            text: 'Chatbot'
            // disabled: true
        }
    ]);
    const isCalling = ref(false);
    const handleChangeLanguage = val => {
        console.log('val: ', val);
        language.value = val;
        locale.value = val;
        localStorage.setItem('language', val);
    };
    const getMenuTab = val => {
        let text = '';
        switch (val) {
            case 'video':
                text = t('menuTabVideo');
                break;
            case 'voice':
                text = t('menuTabAudio');
                break;
            case 'chatbot':
                text = t('menuTabChatbot');
                break;
            default:
                break;
        }
        return text;
    };
    const handleClickTab = (val, index) => {
        activeTab.value = val;
        const port = route.query.port;
        const type = index;
        router.push({
            path: '/',
            query: {
                port,
                type
            }
        });
    };
</script>

<style lang="less" scoped>
    .home-page {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        &-header {
            display: flex;
            align-items: center;
            &-logo {
                width: 174px;
                height: 46px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 12px;
                background: #ffffff;
                flex-shrink: 0;
                padding: 0 24px;
                .logo-icon {
                    width: 100%;
                    height: 100%;
                }
            }
            &-menu {
                display: flex;
                align-items: center;
                margin-left: 16px;
                &-item {
                    width: 260px;
                    height: 46px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: #ffffff;
                    color: #252525;
                    font-family: PingFang SC;
                    font-size: 16px;
                    font-style: normal;
                    font-weight: 400;
                    line-height: normal;
                    border: 1px solid #dde1eb;
                    cursor: pointer;
                    user-select: none;
                }
                &-item + &-item {
                    border-left: none;
                }
                &-item:first-of-type {
                    border-radius: 12px 0 0 12px;
                }
                &-item:last-of-type {
                    border-radius: 0 12px 12px 0;
                }
                .active-tab {
                    color: #ffffff;
                    background: linear-gradient(90deg, #789efe 0.02%, #647fff 75.28%);
                    font-weight: 500;
                }
                .disabled-tab {
                    cursor: not-allowed;
                    border-color: #dde1eb;
                    color: #d1d1d1;
                }
            }
            &-switch {
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                .change-language {
                    display: flex;
                    align-items: center;
                    &-item {
                        width: 80px;
                        height: 32px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        border: 1px solid #dde1eb;
                        background: #ffffff;
                        color: #252525;
                        font-family: PingFang SC;
                        font-size: 14px;
                        font-weight: 400;
                        line-height: normal;
                        cursor: pointer;
                        user-select: none;
                    }
                    &-item:first-of-type {
                        border-right: none;
                        border-radius: 12px 0 0 12px;
                    }
                    &-item:last-of-type {
                        border-radius: 0 12px 12px 0;
                    }
                    &-item.active-language {
                        color: #ffffff;
                        background: linear-gradient(90deg, #789efe 0.02%, #647fff 75.28%);
                    }
                }
            }
        }
        &-content {
            flex: 1;
            height: 0;
            border-radius: 12px;
            margin-top: 16px;
            background: #ffffff;
            padding: 18px;
            display: flex;
            .config-box {
                width: 322px;
                margin-left: 16px;
                // border-left: 1px solid black;
                box-shadow: -0.5px 0 0 0 #e0e0e0;
                overflow: auto;
            }
        }
        .no-padding {
            padding: 0;
            overflow: hidden;
            background: #ffffff;
        }
    }
</style>
<style lang="less">
    .el-popover.el-popper.config-popover {
        padding: 18px;
        border-radius: 12px;
    }
</style>

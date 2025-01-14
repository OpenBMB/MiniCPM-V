<template>
    <div class="like-box">
        <div class="like-btn" @click="selectFeedbackStatus('like')">
            <img v-if="feedbackStatus === '' || feedbackStatus === 'dislike'" src="@/assets/images/zan.png" />
            <img v-else src="@/assets/images/zan-active.png" />
        </div>
        <div class="dislike-btn" @click="selectFeedbackStatus('dislike')">
            <img v-if="feedbackStatus === '' || feedbackStatus === 'like'" src="@/assets/images/cai.png" />
            <img v-else src="@/assets/images/cai-active.png" />
        </div>
    </div>
    <el-dialog
        v-model="dialogVisible"
        :title="t('feedbackDialogTitle')"
        width="400"
        :align-center="true"
        @close="cancelFeedback"
    >
        <el-input type="textarea" :rows="4" v-model="comment" />
        <div class="operate-btn">
            <el-button type="primary" :loading="submitLoading" @click="submitFeedback">确定</el-button>
            <el-button @click="cancelFeedback">取消</el-button>
        </div>
    </el-dialog>
</template>
<script setup>
    import { feedback } from '@/apis';
    import { useI18n } from 'vue-i18n';

    const { t } = useI18n();
    const feedbackStatus = defineModel('feedbackStatus');
    const curResponseId = defineModel('curResponseId');
    const dialogVisible = ref(false);
    const comment = ref('');
    const submitLoading = ref(false);
    const selectFeedbackStatus = val => {
        if (!curResponseId.value) {
            return;
        }
        feedbackStatus.value = val;
        dialogVisible.value = true;
    };
    // 提交反馈
    const submitFeedback = async () => {
        submitLoading.value = true;
        const { code, message } = await feedback({
            response_id: curResponseId.value,
            rating: feedbackStatus.value,
            comment: comment.value
        });
        submitLoading.value = false;
        if (code !== 0) {
            ElMessage({
                type: 'error',
                message: message,
                duration: 3000,
                customClass: 'system-error'
            });
            return;
        }
        ElMessage.success('反馈成功');
        dialogVisible.value = false;
        setTimeout(() => {
            feedbackStatus.value = '';
        }, 2000);
    };
    const cancelFeedback = () => {
        dialogVisible.value = false;
        feedbackStatus.value = '';
    };
</script>
<style lang="less" scoped>
    .like-box {
        display: flex;
        margin: 0 16px;
        .like-btn,
        .dislike-btn {
            width: 26px;
            height: 26px;
            background: #f3f3f3;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            cursor: pointer;
            &:hover {
                background: #d1d1d1;
            }
            img {
                width: 16px;
                height: 16px;
            }
        }
        .dislike-btn {
            margin-left: 16px;
        }
    }
    .operate-btn {
        margin-top: 20px;
        display: flex;
        justify-content: flex-end;
        .el-button--primary {
            background: #647fff;
            border-color: #647fff;
            &:hover {
                border-color: #647fff;
            }
        }
    }
</style>

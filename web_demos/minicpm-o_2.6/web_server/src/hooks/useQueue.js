export class TaskQueue {
    constructor() {
        this.tasks = [];
        this.isRunning = false;
        this.isPaused = false;
        this.currentTask = null;
    }

    // 添加任务到队列
    addTask(task) {
        this.tasks.push(task);
        if (!this.isRunning) {
            this.start();
        }
    }

    // 删除任务
    removeTask(taskToRemove) {
        this.tasks = this.tasks.filter(task => task !== taskToRemove);
    }

    // 清空任务队列
    clearQueue() {
        this.tasks = [];
    }

    // 暂停任务执行
    pause() {
        this.isPaused = true;
    }

    // 恢复任务执行
    resume() {
        if (this.isPaused) {
            this.isPaused = false;
            if (!this.isRunning) {
                this.start();
            }
        }
    }

    // 内部启动方法
    async start() {
        this.isRunning = true;
        while (this.tasks.length > 0 && !this.isPaused) {
            this.currentTask = this.tasks.shift();
            await this.currentTask();

            // 检查是否暂停或任务队列已清空
            if (this.isPaused || this.tasks.length === 0) {
                this.isRunning = false;
                break;
            }
        }
        this.isRunning = false;
    }
}

// 示例任务函数
function exampleTask(id) {
    return () =>
        new Promise(resolve => {
            console.log(`Executing task ${id}`);
            setTimeout(() => {
                console.log(`Task ${id} completed`);
                resolve();
            }, 1000); // 每个任务耗时1秒
        });
}

// 测试示例
const queue = new TaskQueue();

// 添加任务到队列
for (let i = 1; i <= 5; i++) {
    queue.addTask(exampleTask(i));
}

// 暂停队列，在2.5秒后执行
setTimeout(() => {
    console.log('Pausing queue...');
    queue.pause();
}, 2500);

// 恢复队列，在4.5秒后执行
setTimeout(() => {
    console.log('Resuming queue...');
    queue.resume();
}, 4500);

// 清空队列，在3秒后执行
setTimeout(() => {
    console.log('Clearing queue...');
    queue.clearQueue();
}, 3000);

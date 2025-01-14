class WebSocketClient {
    constructor(url, maxReconnectAttempts = 5, reconnectInterval = 5000) {
        this.url = url;
        this.socket = null;
        this.eventHandlers = {};
        this.maxReconnectAttempts = maxReconnectAttempts;
        this.reconnectInterval = reconnectInterval;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
    }

    connect() {
        this.reconnectAttempts = 0;
        this.establishConnection();
    }

    establishConnection() {
        this.socket = new WebSocket(this.url);

        this.socket.onopen = () => {
            console.log('WebSocket connection opened');
            this.reconnectAttempts = 0; // Reset reconnect attempts on successful connection
            this.emit('open');
        };

        this.socket.onclose = event => {
            console.log('WebSocket connection closed', event);
            this.emit('close', event);
            // 1005为主动关闭websocket
            if (event.code !== 1005) {
                this.reconnect();
            }
        };

        this.socket.onerror = error => {
            console.error('WebSocket error', error);
            this.emit('error', error);
            // Optionally, you may want to trigger a reconnect on error as well
            // this.reconnect();
        };

        this.socket.onmessage = message => {
            // console.log('WebSocket message received', message.data);
            this.emit('message', message.data);
        };
    }

    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(data);
        } else {
            console.error('WebSocket is not open');
        }
    }

    on(event, handler) {
        // if (!this.eventHandlers[event]) {
        this.eventHandlers[event] = [];
        // }
        this.eventHandlers[event].push(handler);
        // console.log('Event handler added:', this.eventHandlers, event);
    }

    emit(event, ...args) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => handler(...args));
        }
    }

    close() {
        if (this.socket) {
            this.socket.close();
        }
        clearTimeout(this.reconnectTimer);
    }

    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            console.log(`Reconnecting attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts}...`);
            this.reconnectTimer = setTimeout(() => {
                this.reconnectAttempts++;
                this.establishConnection();
            }, this.reconnectInterval);
        } else {
            console.error('Max reconnect attempts reached. WebSocket will not attempt to reconnect.');
            this.emit('max-reconnect-attempts');
        }
    }
}

export default WebSocketClient;

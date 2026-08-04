# MiniCPM-o 4.5 离线部署实战指南（WSL2 构建镜像 → 上传内网 H100 服务器 → 本地 + 手机访问）

> 目标：你在本地 Windows系统PC + WSL2 构建 Docker 镜像，把镜像和模型传到无公网的公司 H100 服务器，启动服务后在本地浏览器和Android系统手机上测试全双工视频通话。

**你的环境速查：**

| 项目 | 值 |
| --- | --- |
| 服务器 SSH | `ssh -p $SSH_PORT $SSH_USER@$SSH_HOST`（端口可能动态变化） |
| GPU | NVIDIA H100（驱动 550.90.12） |
| CUDA | 12.4（与 Dockerfile 基础镜像 `cuda:12.4.1` 完全匹配） |
| 本地 | Win10 + WSL2 Ubuntu |

**每次执行前先设置 SSH 变量（只改这里即可）：**

```bash
export SSH_HOST=<YOUR_HOST>
export SSH_PORT=<YOUR_PORT>
export SSH_USER=<YOUR_USER>
```

PowerShell 等价写法（Windows 终端直接用）：

```powershell
$env:SSH_HOST = "<YOUR_HOST>"
$env:SSH_PORT = "<YOUR_PORT>"
$env:SSH_USER = "<YOUR_USER>"
```

## PowerShell 日常三命令速查（推荐）

```powershell
# 1) 端口变化时先更新 SSH 参数
Set-MiniCPMSSH -Port "<YOUR_PORT>" -User "<YOUR_USER>"

# 2) 启动手机模式（开隧道 + 打印可访问 URL）
Start-MiniCPMMobile

# 3) 结束隧道
Stop-MiniCPMMobile
```

端口变化后的快速恢复：

```powershell
Set-MiniCPMSSH -Port "<YOUR_PORT>" -User "<YOUR_USER>"
Restart-MiniCPMMobile
```

PowerShell 中引用变量时，`ssh/scp` 建议写成：

```powershell
ssh -p $env:SSH_PORT "$env:SSH_USER@$env:SSH_HOST"
scp -P $env:SSH_PORT .\file.tar.gz "$env:SSH_USER@$env:SSH_HOST:<YOUR_PATH>/deploy_pkg/"
```

可选：定义一个一键函数（以后只改端口即可）

```powershell
function Set-MiniCPMSSH {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Port,
    [string]$Host = "<YOUR_HOST>",
    [string]$User = "<YOUR_USER>"
  )

  $env:SSH_HOST = $Host
  $env:SSH_PORT = $Port
  $env:SSH_USER = $User

  Write-Host "[MiniCPM SSH] HOST=$env:SSH_HOST PORT=$env:SSH_PORT USER=$env:SSH_USER"
}
```

使用示例：

```powershell
Set-MiniCPMSSH -Port "<YOUR_PORT>" -User "<YOUR_USER>"
ssh -p $env:SSH_PORT "$env:SSH_USER@$env:SSH_HOST"
```

可选：定义一键开隧道函数（本机/手机两种模式）

```powershell
function Open-MiniCPMTunnel {
  param(
    [ValidateSet("local", "mobile")]
    [string]$Mode = "local"
  )

  if (-not $env:SSH_HOST -or -not $env:SSH_PORT -or -not $env:SSH_USER) {
    throw "请先执行 Set-MiniCPMSSH 设置 SSH_HOST/SSH_PORT/SSH_USER"
  }

  if ($Mode -eq "local") {
    ssh -N -p $env:SSH_PORT `
      -L 3000:127.0.0.1:3000 `
      -L 3443:127.0.0.1:3443 `
      -L 32550:127.0.0.1:32550 `
      "$env:SSH_USER@$env:SSH_HOST"
  }
  else {
    ssh -N -p $env:SSH_PORT `
      -L 0.0.0.0:3443:127.0.0.1:3443 `
      "$env:SSH_USER@$env:SSH_HOST"
  }
}
```

使用示例：

```powershell
# 1) 设置动态 SSH 参数
Set-MiniCPMSSH -Port "54062" -User "your_user"

# 2) 仅本机访问（浏览器打开 http://127.0.0.1:3000）
Open-MiniCPMTunnel -Mode local

# 3) 手机访问（同一 WiFi，用 https://笔记本局域网IP:3443）
Open-MiniCPMTunnel -Mode mobile
```

可选：自动打印手机访问地址

```powershell
function Get-MiniCPMLanUrl {
  param(
    [int]$Port = 3443
  )

  $ipv4List = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
      $_.IPAddress -notlike '127.*' -and
      $_.IPAddress -notlike '169.254.*' -and
      $_.PrefixOrigin -ne 'WellKnown'
    } |
    Sort-Object -Property InterfaceMetric

  if (-not $ipv4List) {
    throw "未找到可用 IPv4 地址，请检查网卡/网络连接"
  }

  $ip = $ipv4List[0].IPAddress
  $url = "https://$ip`:$Port"

  Write-Host "[MiniCPM LAN URL] $url"
  return $url
}
```

使用示例：

```powershell
# 先开启手机模式隧道（在另一个终端窗口运行）
Open-MiniCPMTunnel -Mode mobile

# 当前窗口打印手机访问地址
Get-MiniCPMLanUrl
```

可选：一键启动手机模式（开隧道 + 检查端口 + 打印 URL）

```powershell
function Start-MiniCPMMobile {
  param(
    [int]$Port = 3443
  )

  if (-not $env:SSH_HOST -or -not $env:SSH_PORT -or -not $env:SSH_USER) {
    throw "请先执行 Set-MiniCPMSSH 设置 SSH_HOST/SSH_PORT/SSH_USER"
  }

  $sshCmd = "ssh -N -p $env:SSH_PORT -L 0.0.0.0:$Port`:127.0.0.1:$Port $env:SSH_USER@$env:SSH_HOST"

  # 在新窗口开隧道，避免阻塞当前终端
  $proc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $sshCmd -PassThru
  $env:MINICPM_MOBILE_SSH_PID = [string]$proc.Id
  $env:MINICPM_MOBILE_PORT = [string]$Port
  Start-Sleep -Seconds 2

  $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if (-not $listener) {
    Write-Warning "未检测到本机 $Port 端口监听，请检查 SSH 是否连接成功。"
    return
  }

  $url = Get-MiniCPMLanUrl -Port $Port
  Write-Host "[MiniCPM Mobile PID] $env:MINICPM_MOBILE_SSH_PID"
  Write-Host "[MiniCPM Mobile Ready] 手机浏览器访问: $url"
}

function Stop-MiniCPMMobile {
  $pidText = $env:MINICPM_MOBILE_SSH_PID

  if ($pidText) {
    $pidValue = [int]$pidText
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($proc) {
      Stop-Process -Id $pidValue -Force
      Write-Host "[MiniCPM Mobile Stopped] 已停止隧道进程 PID=$pidValue"
      Remove-Item Env:MINICPM_MOBILE_SSH_PID -ErrorAction SilentlyContinue
      Remove-Item Env:MINICPM_MOBILE_PORT -ErrorAction SilentlyContinue
      return
    }
  }

  $port = if ($env:MINICPM_MOBILE_PORT) { [int]$env:MINICPM_MOBILE_PORT } else { 3443 }
  $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  if (-not $listeners) {
    Write-Host "[MiniCPM Mobile] 未检测到监听端口 $port，无需停止。"
    return
  }

  foreach ($item in $listeners) {
    if ($item.OwningProcess -gt 0) {
      try {
        Stop-Process -Id $item.OwningProcess -Force -ErrorAction Stop
        Write-Host "[MiniCPM Mobile Stopped] 已停止监听端口 $port 的进程 PID=$($item.OwningProcess)"
      }
      catch {
        Write-Warning "停止 PID=$($item.OwningProcess) 失败：$($_.Exception.Message)"
      }
    }
  }

  Remove-Item Env:MINICPM_MOBILE_SSH_PID -ErrorAction SilentlyContinue
  Remove-Item Env:MINICPM_MOBILE_PORT -ErrorAction SilentlyContinue
}

function Restart-MiniCPMMobile {
  param(
    [int]$Port = 3443
  )

  Stop-MiniCPMMobile
  Start-Sleep -Seconds 1
  Start-MiniCPMMobile -Port $Port
}
```

使用示例：

```powershell
# 1) 先设置动态 SSH 参数（端口变更时只改这里）
Set-MiniCPMSSH -Port "54062" -User "your_user"

# 2) 一键启动手机模式并输出可访问地址
Start-MiniCPMMobile

# 3) 端口变化后，一键重启手机模式（可选）
Restart-MiniCPMMobile

# 4) 结束手机模式隧道
Stop-MiniCPMMobile
```

---

## 0. 目录与文件说明

本指南使用了你仓库中新建的部署文件：

- `deploy/Dockerfile.backend`：后端推理服务镜像（FastAPI + MiniCPM-o 4.5）
- `deploy/Dockerfile.frontend`：前端镜像（Vue build + Nginx）
- `deploy/nginx.docker.conf`：Nginx 反向代理到后端容器
- `deploy/docker-compose.yml`：双容器编排（frontend + backend）
- `deploy/requirements.backend.txt`：后端 Python 依赖清单
- `deploy/gen_ssl_cert.sh`：自签名 SSL 证书生成脚本（手机端 HTTPS 必需）

---

## 1. 本地（WSL2）前置准备

在 WSL2 Ubuntu 执行：

```bash
cd <YOUR_PATH>/MiniCPM-o

# 1) 检查 Docker
sudo docker --version
sudo docker compose version

# 2) 如果你当前用户不能直接用 docker，可先临时用 sudo docker
# 或将用户加入 docker 组（重新登录后生效）
# sudo usermod -aG docker $USER
```

> 说明：本地 1050Ti 不参与推理，本地只负责构建镜像，不需要本地 GPU。

---

## 2. 本地下载模型（用于上传到内网）

推荐在本地（有网环境）下载 HuggingFace 模型，再打包上传。

### 2.1 安装下载工具

```bash
python3 -m pip install -U huggingface_hub
```

### 2.2 下载 MiniCPM-o 4.5

```bash
mkdir -p <YOUR_PATH>/MiniCPM-o/models
python3 - << 'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openbmb/MiniCPM-o-4_5',
    local_dir='<YOUR_PATH>/MiniCPM-o/models/MiniCPM-o-4_5',
    local_dir_use_symlinks=False,
    resume_download=True
)
PY
```

下载后检查体积和关键文件：

```bash
du -sh <YOUR_PATH>/MiniCPM-o/models/MiniCPM-o-4_5
ls -lh <YOUR_PATH>/MiniCPM-o/models/MiniCPM-o-4_5 | head
```

---

## 3. 在 WSL2 构建两个镜像

在仓库根目录执行：

```bash
cd <YOUR_PATH>/MiniCPM-o

# 后端镜像
docker build -f deploy/Dockerfile.backend -t minicpmo-backend:latest .

# 前端镜像
docker build -f deploy/Dockerfile.frontend -t minicpmo-frontend:latest .
```

验证镜像存在：

```bash
docker images | grep minicpmo
```

---

## 4. 导出镜像 + 生成 SSL 证书

### 4.1 导出镜像为 tar

```bash
mkdir -p <YOUR_PATH>/deploy_pkg

docker save -o <YOUR_PATH>/deploy_pkg/minicpmo-backend_latest.tar minicpmo-backend:latest
docker save -o <YOUR_PATH>/deploy_pkg/minicpmo-frontend_latest.tar minicpmo-frontend:latest

# 打包 compose 与 nginx 配置
cp deploy/docker-compose.yml <YOUR_PATH>/deploy_pkg/
cp deploy/nginx.docker.conf <YOUR_PATH>/deploy_pkg/
```

可选：压缩减少传输体积

```bash
cd <YOUR_PATH>/deploy_pkg
gzip -1 minicpmo-backend_latest.tar
gzip -1 minicpmo-frontend_latest.tar
```

### 4.2 生成自签名 SSL 证书（手机端 HTTPS 必需）

```bash
cd <YOUR_PATH>/MiniCPM-o
bash deploy/gen_ssl_cert.sh <YOUR_PATH>/deploy_pkg/certs
```

这会在 `<YOUR_PATH>/deploy_pkg/certs/` 下生成 `server.crt` 和 `server.key`。

---

## 5. 上传到内网服务器

你已经通过公司内网认证，且端口可能动态变化，请使用上面定义的 SSH 变量。

### 5.1 上传镜像包和配置文件

```bash
# 先在服务器上创建目标目录
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p <YOUR_PATH>/deploy_pkg"

# 上传镜像 tar 包
scp -P $SSH_PORT -o ServerAliveInterval=60 \
    <YOUR_PATH>/deploy_pkg/minicpmo-backend_latest.tar.gz \
    <YOUR_PATH>/deploy_pkg/minicpmo-frontend_latest.tar.gz \
    <YOUR_PATH>/deploy_pkg/docker-compose.yml \
    <YOUR_PATH>/deploy_pkg/nginx.docker.conf \
  $SSH_USER@$SSH_HOST:<YOUR_PATH>/deploy_pkg/
```

### 5.2 上传模型权重

```bash
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p <YOUR_PATH>/models"

scp -P $SSH_PORT -r -o ServerAliveInterval=60 \
    <YOUR_PATH>/MiniCPM-o/models/MiniCPM-o-4_5 \
  $SSH_USER@$SSH_HOST:<YOUR_PATH>/models/
```

### 5.3 上传 SSL 证书（手机端访问需要）

```bash
scp -P $SSH_PORT -r <YOUR_PATH>/deploy_pkg/certs \
  $SSH_USER@$SSH_HOST:<YOUR_PATH>/deploy_pkg/
```

> 如果端口变更，只需要修改 `SSH_PORT` 变量并重试命令。

---

## 6. H100 服务器准备（一次性）

通过已建立的隧道登录服务器：

```bash
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST
```

检查环境：

```bash
# 确认 NVIDIA 驱动（你已确认: 550.90.12, CUDA 12.4 ✓）
nvidia-smi

# 检查 Docker
docker --version
docker compose version
```

### 6.1 安装 NVIDIA Container Toolkit（若未安装）

如果 `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 失败，需要安装 toolkit。

安装后重启 Docker：

```bash
sudo systemctl restart docker
```

再验证：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 7. H100 服务器加载镜像与启动服务

在服务器上执行（通过 `ssh -p $SSH_PORT $SSH_USER@$SSH_HOST` 登录后）：

```bash
cd /data/minicpmo/deploy_pkg

# 若上传的是 .tar.gz，先解压
gunzip -f minicpmo-backend_latest.tar.gz || true
gunzip -f minicpmo-frontend_latest.tar.gz || true

# 加载镜像
docker load -i minicpmo-backend_latest.tar
docker load -i minicpmo-frontend_latest.tar

# 放置运行时文件
mkdir -p /data/minicpmo/runtime/certs
cp docker-compose.yml /data/minicpmo/runtime/
cp certs/server.crt certs/server.key /data/minicpmo/runtime/certs/

cd /data/minicpmo/runtime
```

### 7.1 设置模型路径并启动

`docker-compose.yml` 里用了 `MODEL_PATH` 环境变量。你可以直接导出：

```bash
export MODEL_PATH=<YOUR_PATH>/models/MiniCPM-o-4_5
export CERTS_PATH=./certs
export BACKEND_PORT=32550

# 兼容两种 Compose 命令：docker compose / docker-compose
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "未找到 Compose，请先安装 docker-compose 或 docker compose 插件" && exit 1
fi

$COMPOSE_CMD -f docker-compose.yml up -d
```

如果两种 Compose 都不可用（`docker compose` / `docker-compose` 都不存在），可直接用 `docker run` 启动：

```bash
docker network create minicpmo-net || true
docker rm -f minicpmo-backend minicpmo-frontend 2>/dev/null || true

docker run -d \
  --name minicpmo-backend \
  --restart unless-stopped \
  --gpus all \
  -e BACKEND_PORT=${BACKEND_PORT:-32550} \
  -p ${BACKEND_PORT:-32550}:${BACKEND_PORT:-32550} \
  -v ${MODEL_PATH}:/models/MiniCPM-o-4_5:ro \
  --network minicpmo-net \
  minicpmo-backend:latest

docker run -d \
  --name minicpmo-frontend \
  --restart unless-stopped \
  -e BACKEND_PORT=${BACKEND_PORT:-32550} \
  -p 3000:3000 \
  -p 3443:3443 \
  -v ${CERTS_PATH}:/etc/nginx/certs:ro \
  --network minicpmo-net \
  minicpmo-frontend:latest
```

如果出现 `Failed to Setup IP tables` 或 `No chain/target/match by that name`，可先绕过 bridge 网络，改用 `host` 网络启动：

```bash
docker rm -f minicpmo-backend minicpmo-frontend 2>/dev/null || true

docker run -d \
  --name minicpmo-backend \
  --restart unless-stopped \
  --gpus all \
  --network host \
  -e BACKEND_PORT=${BACKEND_PORT:-32550} \
  -v ${MODEL_PATH}:/models/MiniCPM-o-4_5:ro \
  minicpmo-backend:latest

docker run -d \
  --name minicpmo-frontend \
  --restart unless-stopped \
  --network host \
  --add-host model-backend:127.0.0.1 \
  -e BACKEND_PORT=${BACKEND_PORT:-32550} \
  -v ${CERTS_PATH}:/etc/nginx/certs:ro \
  minicpmo-frontend:latest
```

查看状态：

```bash
if [ -z "$COMPOSE_CMD" ]; then
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
  else
    COMPOSE_CMD="docker-compose"
  fi
fi

$COMPOSE_CMD -f docker-compose.yml ps
docker logs -f minicpmo-backend
```

若使用 `docker run` 方案，查看状态命令：

```bash
docker ps --filter name=minicpmo
docker logs -f minicpmo-backend
```

健康检查：

```bash
curl http://127.0.0.1:32550/api/v1/health
```

应返回：

```json
{"status":"OK"}
```

> 首次加载模型会较慢（几十秒到数分钟），日志出现模型初始化完成后再测试前端。

---

## 8. 本地电脑访问（SSH 端口转发）

你已能连通 SSH 隧道，只需基于当前端口做服务转发。

在本地 PowerShell 或 WSL 新开一个终端：

```bash
ssh -N -p $SSH_PORT \
  -L 3000:127.0.0.1:3000 \
  -L 3443:127.0.0.1:3443 \
  -L 32550:127.0.0.1:32550 \
  $SSH_USER@$SSH_HOST
```

保持该终端不断开。然后在本地浏览器访问：

- 前端页面（HTTP）：<http://127.0.0.1:3000>
- 前端页面（HTTPS）：<https://127.0.0.1:3443>（自签名证书，需点击"继续前往"）
- 后端健康检查：<http://127.0.0.1:32550/api/v1/health>

> 浏览器会请求摄像头/麦克风权限，点击允许。本地用 `localhost` 访问时 HTTP 即可获取摄像头权限。

---

## 9. 手机端访问（全双工视频通话）

### 9.1 问题与原理

手机浏览器（Chrome/Safari）要调用摄像头和麦克风，**必须使用 HTTPS**（`localhost` 例外，但手机并非 localhost）。

方案：**笔记本做中继** — 手机 → 笔记本 WiFi 局域网 IP → SSH 隧道 → 服务器。

```text
手机浏览器 ──WiFi──▶ 笔记本:3443 ──SSH隧道──▶ H100:3443 ──Nginx──▶ 后端:32550
  (HTTPS)              (绑定 0.0.0.0)
```

### 9.2 操作步骤

#### Step 1：建立"全接口绑定"的 SSH 隧道

```bash
ssh -N -p $SSH_PORT \
  -L 0.0.0.0:3443:127.0.0.1:3443 \
  $SSH_USER@$SSH_HOST
```

> 关键区别：`0.0.0.0:3443` 让笔记本的所有网卡都监听 3443 端口，同一 WiFi 的手机才能连入。

#### Step 2：查看笔记本局域网 IP

PowerShell 中执行：

```powershell
ipconfig | Select-String "IPv4"
```

假设得到 `192.168.1.100`。

#### Step 3：Windows 防火墙放行端口

PowerShell（管理员）执行：

```powershell
New-NetFirewallRule -DisplayName "MiniCPMo HTTPS" -Direction Inbound -LocalPort 3443 -Protocol TCP -Action Allow
```

#### Step 4：手机浏览器访问

确保手机与笔记本连同一 WiFi，然后在手机浏览器输入：

```text
https://192.168.1.100:3443
```

- **首次访问**会提示"不安全连接"（自签名证书），选择 **「高级」→「继续前往」**
- 接着浏览器会请求摄像头/麦克风权限，**允许**即可
- 进入视频通话页面，开始全双工对话

### 9.3 iOS Safari 注意事项

iOS Safari 对自签名证书更严格。如果无法通过上述方式跳过：

1. 在手机上用 Safari 打开 `https://192.168.1.100:3443/certs/server.crt`（若你配置了证书下载路径），下载安装证书
2. 或者将 `server.crt` 通过 AirDrop / 微信发送到手机，在 **设置 → 通用 → 描述文件 → 安装**
3. 再到 **设置 → 通用 → 关于本机 → 证书信任设置 → 启用完全信任**

之后 Safari 访问 `https://192.168.1.100:3443` 即可正常使用。

---

## 10. 常见问题与排查

### 10.1 前端能打开，但无法对话

检查后端日志：

```bash
docker logs --tail 200 minicpmo-backend
```

重点看：

- 模型路径是否存在：`/models/MiniCPM-o-4_5`
- 显存是否足够（H100 通常充足）
- 是否出现 `trust_remote_code` / 依赖版本错误

### 10.2 容器内 GPU 不可见

```bash
docker exec -it minicpmo-backend nvidia-smi
```

若失败，优先检查 NVIDIA Container Toolkit 与 Docker daemon 配置。

### 10.3 WebSocket / SSE 异常

本项目已在 `nginx.docker.conf` 关闭缓冲并配置了 websocket upgrade。
若仍异常，检查公司内网网关是否拦截长连接。

### 10.4 模型启动太慢

首次启动可能较慢；后续会快很多。可先看：

```bash
nvidia-smi
docker logs -f minicpmo-backend
```

---

## 11. 你下一步可以做的优化（可选）

1. 将后端镜像改为“离线 wheel 安装模式”，彻底避免服务器 pip 联网需求。  
2. 使用私有镜像仓库（Harbor）替代 tar 包传输。  
3. 用 systemd 或 cron 做容器自动拉起与日志轮转。  
4. 替换自签名证书为企业 CA 签发的证书，手机端免手动信任。

---

## 12. 一键启动命令速查

### H100 侧（假设文件已上传）

```bash
cd <YOUR_PATH>/deploy_pkg

docker load -i minicpmo-backend_latest.tar
docker load -i minicpmo-frontend_latest.tar

mkdir -p <YOUR_PATH>/runtime/certs
cp docker-compose.yml <YOUR_PATH>/runtime/
cp certs/server.* <YOUR_PATH>/runtime/certs/

cd <YOUR_PATH>/runtime
export MODEL_PATH=<YOUR_PATH>/models/MiniCPM-o-4_5
export CERTS_PATH=./certs
export BACKEND_PORT=32550
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "未找到 Compose，请先安装 docker-compose 或 docker compose 插件" && exit 1
fi

$COMPOSE_CMD -f docker-compose.yml up -d
```

### 本地电脑（开隧道）

```bash
ssh -N -p $SSH_PORT -L 3000:127.0.0.1:3000 -L 3443:127.0.0.1:3443 -L 32550:127.0.0.1:32550 $SSH_USER@$SSH_HOST
```

PowerShell 版本：

```powershell
ssh -N -p $env:SSH_PORT -L 3000:127.0.0.1:3000 -L 3443:127.0.0.1:3443 -L 32550:127.0.0.1:32550 "$env:SSH_USER@$env:SSH_HOST"
```

本地电脑打开：<http://127.0.0.1:3000>

### 手机端（通过笔记本中转）

```bash
# 笔记本绑定所有网卡
ssh -N -p $SSH_PORT -L 0.0.0.0:3443:127.0.0.1:3443 $SSH_USER@$SSH_HOST
```

手机浏览器打开：`https://<YOUR_LAPTOP_LAN_IP>:3443`

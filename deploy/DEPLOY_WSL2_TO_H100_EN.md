# MiniCPM-o 4.5 Offline Deployment Guide (Build Image in WSL2 → Upload to Intranet H100 Server → Local + Mobile Access)

> Goal: Build a Docker image on your local Windows PC with WSL2, transfer the image and model to a company H100 server with no public internet access, start the service, and test full-duplex video calling in a local browser and on an Android phone.

**Your Environment Quick Reference:**

| Item | Value |
| --- | --- |
| Server SSH | `ssh -p $SSH_PORT $SSH_USER@$SSH_HOST` (port may change dynamically) |
| GPU | NVIDIA H100 (driver 550.90.12) |
| CUDA | 12.4 (fully matches the Dockerfile base image `cuda:12.4.1`) |
| Local | Win10 + WSL2 Ubuntu |

**Set SSH variables before each run (only change here):**

```bash
export SSH_HOST=127.0.0.1
export SSH_PORT=54062
export SSH_USER=your_user
```

PowerShell equivalent (use directly in Windows Terminal):

```powershell
$env:SSH_HOST = "127.0.0.1"
$env:SSH_HOST = "<YOUR_HOST>"
$env:SSH_PORT = "<YOUR_PORT>"
$env:SSH_USER = "<YOUR_USER>"

## PowerShell Daily Three-Command Quick Reference (Recommended)

```powershell
Set-MiniCPMSSH -Port "<YOUR_PORT>" -User "<YOUR_USER>"
# 1) Update SSH parameters when port changes
Set-MiniCPMSSH -Port "54062" -User "your_user"

# 2) Start mobile mode (open tunnel + print accessible URL)
Set-MiniCPMSSH -Port "<YOUR_PORT>" -User "<YOUR_USER>"
Start-MiniCPMMobile

# 3) Stop tunnel
Stop-MiniCPMMobile
scp -P $env:SSH_PORT .\file.tar.gz "$env:SSH_USER@$env:SSH_HOST:<YOUR_PATH>/deploy_pkg/"

Quick recovery after port change:

    [string]$Host = "<YOUR_HOST>",
    [string]$User = "<YOUR_USER>"
Restart-MiniCPMMobile
```

  $env:SSH_HOST = $Host
  $env:SSH_PORT = $Port
  $env:SSH_USER = $User
ssh -p $env:SSH_PORT "$env:SSH_USER@$env:SSH_HOST"
scp -P $env:SSH_PORT .\file.tar.gz "$env:SSH_USER@$env:SSH_HOST:/data/minicpmo/deploy_pkg/"
```
  Write-Host "[MiniCPM SSH] HOST=$env:SSH_HOST PORT=$env:SSH_PORT USER=$env:SSH_USER"
Optional: Define a one-click function (only change the port going forward)

```powershell
function Set-MiniCPMSSH {
Set-MiniCPMSSH -Port "<YOUR_PORT>" -User "<YOUR_USER>"
  param(
    [Parameter(Mandatory = $true)]
    [string]$Port,
    [string]$Host = "127.0.0.1",
Open-MiniCPMTunnel -Mode local
    [string]$User = "your_user"
  )

  $env:SSH_HOST = $Host
Open-MiniCPMTunnel -Mode mobile
  $env:SSH_PORT = $Port
  $env:SSH_USER = $User

  Write-Host "[MiniCPM SSH] HOST=$env:SSH_HOST PORT=$env:SSH_PORT USER=$env:SSH_USER"
ssh -N -p $env:SSH_PORT `
      -L 3000:127.0.0.1:3000 `
      -L 3443:127.0.0.1:3443 `
      -L 32550:127.0.0.1:32550 `
      "$env:SSH_USER@$env:SSH_HOST"

```powershell
Set-MiniCPMSSH -Port "54062" -User "your_user"
ssh -p $env:SSH_PORT "$env:SSH_USER@$env:SSH_HOST"
      -L 0.0.0.0:3443:127.0.0.1:3443 `
      "$env:SSH_USER@$env:SSH_HOST"
Optional: Define a one-click tunnel function (local / mobile modes)

```powershell
function Open-MiniCPMTunnel {
cd <YOUR_PATH>/MiniCPM-o
  param(
    [ValidateSet("local", "mobile")]
    [string]$Mode = "local"
  )
mkdir -p <YOUR_PATH>/MiniCPM-o/models

  if (-not $env:SSH_HOST -or -not $env:SSH_PORT -or -not $env:SSH_USER) {
    throw "Please run Set-MiniCPMSSH first to set SSH_HOST/SSH_PORT/SSH_USER"
  }

du -sh <YOUR_PATH>/MiniCPM-o/models/MiniCPM-o-4_5
ls -lh <YOUR_PATH>/MiniCPM-o/models/MiniCPM-o-4_5 | head
  if ($Mode -eq "local") {
    ssh -N -p $env:SSH_PORT `
      -L 3000:127.0.0.1:3000 `
      -L 3443:127.0.0.1:3443 `
cd <YOUR_PATH>/MiniCPM-o
      -L 32550:127.0.0.1:32550 `
      "$env:SSH_USER@$env:SSH_HOST"
  }
  else {
mkdir -p <YOUR_PATH>/deploy_pkg
    ssh -N -p $env:SSH_PORT `
      -L 0.0.0.0:3443:127.0.0.1:3443 `
      "$env:SSH_USER@$env:SSH_HOST"
  }
}
docker save -o <YOUR_PATH>/deploy_pkg/minicpmo-backend_latest.tar minicpmo-backend:latest
docker save -o <YOUR_PATH>/deploy_pkg/minicpmo-frontend_latest.tar minicpmo-frontend:latest
```

Usage example:

```powershell
cp deploy/docker-compose.yml <YOUR_PATH>/deploy_pkg/
cp deploy/nginx.docker.conf <YOUR_PATH>/deploy_pkg/
# 1) Set dynamic SSH parameters
Set-MiniCPMSSH -Port "54062" -User "your_user"

# 2) Local access only (open http://127.0.0.1:3000 in browser)
cd <YOUR_PATH>/deploy_pkg
Open-MiniCPMTunnel -Mode local

# 3) Mobile access (same WiFi, use https://laptop_lan_ip:3443)
Open-MiniCPMTunnel -Mode mobile
```
gzip -1 minicpmo-backend_latest.tar
gzip -1 minicpmo-frontend_latest.tar

Optional: Auto-print mobile access URL

```powershell
function Get-MiniCPMLanUrl {
cd <YOUR_PATH>/MiniCPM-o
bash deploy/gen_ssl_cert.sh <YOUR_PATH>/deploy_pkg/certs
  param(
    [int]$Port = 3443
  )

This will generate `server.crt` and `server.key` under `<YOUR_PATH>/deploy_pkg/certs/`.
  $ipv4List = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
      $_.IPAddress -notlike '127.*' -and
      $_.IPAddress -notlike '169.254.*' -and
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p <YOUR_PATH>/deploy_pkg"
      $_.PrefixOrigin -ne 'WellKnown'
    } |
    Sort-Object -Property InterfaceMetric

scp -P $SSH_PORT -o ServerAliveInterval=60 \
    <YOUR_PATH>/deploy_pkg/minicpmo-backend_latest.tar.gz \
    <YOUR_PATH>/deploy_pkg/minicpmo-frontend_latest.tar.gz \
    <YOUR_PATH>/deploy_pkg/docker-compose.yml \
    <YOUR_PATH>/deploy_pkg/nginx.docker.conf \
  $SSH_USER@$SSH_HOST:<YOUR_PATH>/deploy_pkg/
  $url = "https://$ip`:$Port"

  Write-Host "[MiniCPM LAN URL] $url"
  return $url
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p <YOUR_PATH>/models"
}
```

Usage example:
scp -P $SSH_PORT -r -o ServerAliveInterval=60 \
    <YOUR_PATH>/MiniCPM-o/models/MiniCPM-o-4_5 \
  $SSH_USER@$SSH_HOST:<YOUR_PATH>/models/
# Start mobile mode tunnel first (run in another terminal window)
Open-MiniCPMTunnel -Mode mobile

# Print mobile access URL in the current window
  $SSH_USER@$SSH_HOST:<YOUR_PATH>/deploy_pkg/
```

Optional: One-click mobile mode startup (open tunnel + check port + print URL)

```powershell
function Start-MiniCPMMobile {
export MODEL_PATH=<YOUR_PATH>/models/MiniCPM-o-4_5
export CERTS_PATH=./certs
export BACKEND_PORT=32550
  param(
    [int]$Port = 3443
  )

  -v ${MODEL_PATH}:/models/MiniCPM-o-4_5:ro \
  if (-not $env:SSH_HOST -or -not $env:SSH_PORT -or -not $env:SSH_USER) {
    throw "Please run Set-MiniCPMSSH first to set SSH_HOST/SSH_PORT/SSH_USER"
  }

  -v ${CERTS_PATH}:/etc/nginx/certs:ro \
  $sshCmd = "ssh -N -p $env:SSH_PORT -L 0.0.0.0:$Port`:127.0.0.1:$Port $env:SSH_USER@$env:SSH_HOST"

  # Open tunnel in a new window to avoid blocking the current terminal
  $proc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $sshCmd -PassThru
  -v ${MODEL_PATH}:/models/MiniCPM-o-4_5:ro \
  $env:MINICPM_MOBILE_SSH_PID = [string]$proc.Id
  $env:MINICPM_MOBILE_PORT = [string]$Port
  Start-Sleep -Seconds 2

  -v ${CERTS_PATH}:/etc/nginx/certs:ro \
  $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if (-not $listener) {
    Write-Warning "No listener detected on local port $Port. Please check whether SSH connected successfully."
    return
cd <YOUR_PATH>/deploy_pkg
  }

  $url = Get-MiniCPMLanUrl -Port $Port
  Write-Host "[MiniCPM Mobile PID] $env:MINICPM_MOBILE_SSH_PID"
  Write-Host "[MiniCPM Mobile Ready] Open on mobile browser: $url"
}
mkdir -p <YOUR_PATH>/runtime/certs
cp docker-compose.yml <YOUR_PATH>/runtime/
cp certs/server.* <YOUR_PATH>/runtime/certs/

function Stop-MiniCPMMobile {
  $pidText = $env:MINICPM_MOBILE_SSH_PID

cd <YOUR_PATH>/runtime
  if ($pidText) {
    $pidValue = [int]$pidText
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($proc) {
      Stop-Process -Id $pidValue -Force
      Write-Host "[MiniCPM Mobile Stopped] Tunnel process stopped PID=$pidValue"
export MODEL_PATH=<YOUR_PATH>/models/MiniCPM-o-4_5
export CERTS_PATH=./certs
export BACKEND_PORT=32550
      Remove-Item Env:MINICPM_MOBILE_SSH_PID -ErrorAction SilentlyContinue
      Remove-Item Env:MINICPM_MOBILE_PORT -ErrorAction SilentlyContinue
      return
    }
ssh -N -p $SSH_PORT -L 3000:127.0.0.1:3000 -L 3443:127.0.0.1:3443 -L 32550:127.0.0.1:32550 $SSH_USER@$SSH_HOST
  }

  $port = if ($env:MINICPM_MOBILE_PORT) { [int]$env:MINICPM_MOBILE_PORT } else { 3443 }
  $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
ssh -N -p $env:SSH_PORT -L 3000:127.0.0.1:3000 -L 3443:127.0.0.1:3443 -L 32550:127.0.0.1:32550 "$env:SSH_USER@$env:SSH_HOST"
  if (-not $listeners) {
    Write-Host "[MiniCPM Mobile] No listener detected on port $port. Nothing to stop."
    return
  }
ssh -N -p $SSH_PORT -L 0.0.0.0:3443:127.0.0.1:3443 $SSH_USER@$SSH_HOST

  foreach ($item in $listeners) {
    if ($item.OwningProcess -gt 0) {
      try {
Open on phone browser: `https://<YOUR_LAPTOP_LAN_IP>:3443`
        Stop-Process -Id $item.OwningProcess -Force -ErrorAction Stop
        Write-Host "[MiniCPM Mobile Stopped] Stopped process listening on port $port PID=$($item.OwningProcess)"
      }
      catch {
        Write-Warning "Failed to stop PID=$($item.OwningProcess): $($_.Exception.Message)"
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

Usage example:

```powershell
# 1) Set dynamic SSH parameters first (only change here when port changes)
Set-MiniCPMSSH -Port "54062" -User "your_user"

# 2) One-click start mobile mode and output the accessible URL
Start-MiniCPMMobile

# 3) One-click restart mobile mode after port change (optional)
Restart-MiniCPMMobile

# 4) Stop mobile mode tunnel
Stop-MiniCPMMobile
```

---

## 0. Directory and File Overview

This guide uses the newly created deployment files in your repository:

- `deploy/Dockerfile.backend`: Backend inference service image (FastAPI + MiniCPM-o 4.5)
- `deploy/Dockerfile.frontend`: Frontend image (Vue build + Nginx)
- `deploy/nginx.docker.conf`: Nginx reverse proxy to backend container
- `deploy/docker-compose.yml`: Two-container orchestration (frontend + backend)
- `deploy/requirements.backend.txt`: Backend Python dependency list
- `deploy/gen_ssl_cert.sh`: Self-signed SSL certificate generation script (required for mobile HTTPS)

---

## 1. Local (WSL2) Prerequisites

Run in WSL2 Ubuntu:

```bash
cd /mnt/d/JiuTian/codes/MiniCPM-o

# 1) Check Docker
sudo docker --version
sudo docker compose version

# 2) If your current user cannot use docker directly, you can temporarily use sudo docker
# Or add the user to the docker group (takes effect after re-login)
# sudo usermod -aG docker $USER
```

> Note: The local 1050Ti does not participate in inference. The local machine is only responsible for building images and does not require a local GPU.

---

## 2. Download the Model Locally (for Upload to Intranet)

It is recommended to download the HuggingFace model locally (where internet is available), then package and upload it.

### 2.1 Install Download Tool

```bash
python3 -m pip install -U huggingface_hub
```

### 2.2 Download MiniCPM-o 4.5

```bash
mkdir -p /mnt/d/JiuTian/codes/MiniCPM-o/models
python3 - << 'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openbmb/MiniCPM-o-4_5',
    local_dir='/mnt/d/JiuTian/codes/MiniCPM-o/models/MiniCPM-o-4_5',
    local_dir_use_symlinks=False,
    resume_download=True
)
PY
```

After downloading, check the size and key files:

```bash
du -sh /mnt/d/JiuTian/codes/MiniCPM-o/models/MiniCPM-o-4_5
ls -lh /mnt/d/JiuTian/codes/MiniCPM-o/models/MiniCPM-o-4_5 | head
```

---

## 3. Build Two Images in WSL2

Run from the repository root directory:

```bash
cd /mnt/d/JiuTian/codes/MiniCPM-o

# Backend image
docker build -f deploy/Dockerfile.backend -t minicpmo-backend:latest .

# Frontend image
docker build -f deploy/Dockerfile.frontend -t minicpmo-frontend:latest .
```

Verify the images exist:

```bash
docker images | grep minicpmo
```

---

## 4. Export Images + Generate SSL Certificate

### 4.1 Export Images as tar

```bash
mkdir -p /mnt/d/JiuTian/deploy_pkg

docker save -o /mnt/d/JiuTian/deploy_pkg/minicpmo-backend_latest.tar minicpmo-backend:latest
docker save -o /mnt/d/JiuTian/deploy_pkg/minicpmo-frontend_latest.tar minicpmo-frontend:latest

# Package compose and nginx config
cp deploy/docker-compose.yml /mnt/d/JiuTian/deploy_pkg/
cp deploy/nginx.docker.conf /mnt/d/JiuTian/deploy_pkg/
```

Optional: Compress to reduce transfer size

```bash
cd /mnt/d/JiuTian/deploy_pkg
gzip -1 minicpmo-backend_latest.tar
gzip -1 minicpmo-frontend_latest.tar
```

### 4.2 Generate Self-Signed SSL Certificate (Required for Mobile HTTPS)

```bash
cd /mnt/d/JiuTian/codes/MiniCPM-o
bash deploy/gen_ssl_cert.sh /mnt/d/JiuTian/deploy_pkg/certs
```

This will generate `server.crt` and `server.key` under `/mnt/d/JiuTian/deploy_pkg/certs/`.

---

## 5. Upload to the Intranet Server

You have already passed company intranet authentication, and the port may change dynamically. Please use the SSH variables defined above.

### 5.1 Upload Image Packages and Config Files

```bash
# First create the target directory on the server
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p /data/minicpmo/deploy_pkg"

# Upload image tar packages
scp -P $SSH_PORT -o ServerAliveInterval=60 \
    /mnt/d/JiuTian/deploy_pkg/minicpmo-backend_latest.tar.gz \
    /mnt/d/JiuTian/deploy_pkg/minicpmo-frontend_latest.tar.gz \
    /mnt/d/JiuTian/deploy_pkg/docker-compose.yml \
    /mnt/d/JiuTian/deploy_pkg/nginx.docker.conf \
  $SSH_USER@$SSH_HOST:/data/minicpmo/deploy_pkg/
```

### 5.2 Upload Model Weights

```bash
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p /data/models"

scp -P $SSH_PORT -r -o ServerAliveInterval=60 \
    /mnt/d/JiuTian/codes/MiniCPM-o/models/MiniCPM-o-4_5 \
  $SSH_USER@$SSH_HOST:/data/models/
```

### 5.3 Upload SSL Certificate (Required for Mobile Access)

```bash
scp -P $SSH_PORT -r /mnt/d/JiuTian/deploy_pkg/certs \
  $SSH_USER@$SSH_HOST:/data/minicpmo/deploy_pkg/
```

> If the port changes, simply update the `SSH_PORT` variable and retry the command.

---

## 6. H100 Server Preparation (One-Time)

Log in to the server through the established tunnel:

```bash
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST
```

Check the environment:

```bash
# Confirm NVIDIA driver (already confirmed: 550.90.12, CUDA 12.4 ✓)
nvidia-smi

# Check Docker
docker --version
docker compose version
```

### 6.1 Install NVIDIA Container Toolkit (If Not Installed)

If `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` fails, you need to install the toolkit.

Restart Docker after installation:

```bash
sudo systemctl restart docker
```

Verify again:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 7. Load Images and Start Services on the H100 Server

Run on the server (after logging in via `ssh -p $SSH_PORT $SSH_USER@$SSH_HOST`):

```bash
cd /data/minicpmo/deploy_pkg

# If the uploaded files are .tar.gz, decompress first
gunzip -f minicpmo-backend_latest.tar.gz || true
gunzip -f minicpmo-frontend_latest.tar.gz || true

# Load images
docker load -i minicpmo-backend_latest.tar
docker load -i minicpmo-frontend_latest.tar

# Place runtime files
mkdir -p /data/minicpmo/runtime/certs
cp docker-compose.yml /data/minicpmo/runtime/
cp certs/server.crt certs/server.key /data/minicpmo/runtime/certs/

cd /data/minicpmo/runtime
```

### 7.1 Set Model Path and Start

`docker-compose.yml` uses the `MODEL_PATH` environment variable. You can export it directly:

```bash
export MODEL_PATH=/data/models/MiniCPM-o-4_5
export CERTS_PATH=./certs
export BACKEND_PORT=32550

# Compatible with both Compose commands: docker compose / docker-compose
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "Compose not found. Please install docker-compose or the docker compose plugin first." && exit 1
fi

$COMPOSE_CMD -f docker-compose.yml up -d
```

If neither Compose option is available (`docker compose` / `docker-compose` both absent), you can start directly with `docker run`:

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

If you encounter `Failed to Setup IP tables` or `No chain/target/match by that name`, you can bypass the bridge network and start with the `host` network instead:

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

Check status:

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

If using the `docker run` approach, check status with:

```bash
docker ps --filter name=minicpmo
docker logs -f minicpmo-backend
```

Health check:

```bash
curl http://127.0.0.1:32550/api/v1/health
```

Expected response:

```json
{"status":"OK"}
```

> The first model load may be slow (tens of seconds to a few minutes). Wait until the logs show model initialization complete before testing the frontend.

---

## 8. Local Computer Access (SSH Port Forwarding)

You can already connect via SSH tunnel — just forward the service ports using the current port.

Open a new terminal in local PowerShell or WSL:

```bash
ssh -N -p $SSH_PORT \
  -L 3000:127.0.0.1:3000 \
  -L 3443:127.0.0.1:3443 \
  -L 32550:127.0.0.1:32550 \
  $SSH_USER@$SSH_HOST
```

Keep this terminal connected. Then open in a local browser:

- Frontend (HTTP): <http://127.0.0.1:3000>
- Frontend (HTTPS): <https://127.0.0.1:3443> (self-signed cert, click "Continue" to proceed)
- Backend health check: <http://127.0.0.1:32550/api/v1/health>

> The browser will request camera/microphone permissions — click Allow. When accessing via `localhost`, HTTP is sufficient to obtain camera permissions.

---

## 9. Mobile Access (Full-Duplex Video Calling)

### 9.1 Problem and Principle

Mobile browsers (Chrome/Safari) **must use HTTPS** to access the camera and microphone (`localhost` is an exception, but the phone is not localhost).

Solution: **Use the laptop as a relay** — Phone → Laptop WiFi LAN IP → SSH tunnel → Server.

```text
Mobile browser ──WiFi──▶ Laptop:3443 ──SSH tunnel──▶ H100:3443 ──Nginx──▶ Backend:32550
   (HTTPS)               (bound to 0.0.0.0)
```

### 9.2 Steps

#### Step 1: Establish an SSH Tunnel with All-Interface Binding

```bash
ssh -N -p $SSH_PORT \
  -L 0.0.0.0:3443:127.0.0.1:3443 \
  $SSH_USER@$SSH_HOST
```

> Key difference: `0.0.0.0:3443` makes all network interfaces on the laptop listen on port 3443, allowing phones on the same WiFi to connect.

#### Step 2: Find the Laptop's LAN IP

Run in PowerShell:

```powershell
ipconfig | Select-String "IPv4"
```

Assume the result is `192.168.1.100`.

#### Step 3: Allow Port Through Windows Firewall

Run in PowerShell (as Administrator):

```powershell
New-NetFirewallRule -DisplayName "MiniCPMo HTTPS" -Direction Inbound -LocalPort 3443 -Protocol TCP -Action Allow
```

#### Step 4: Access from Mobile Browser

Make sure the phone and laptop are on the same WiFi, then enter in the mobile browser:

```text
https://192.168.1.100:3443
```

- **First visit** will show an "unsafe connection" warning (self-signed cert) — tap **"Advanced" → "Continue"**
- The browser will then request camera/microphone permissions — tap **Allow**
- Enter the video call page and start a full-duplex conversation

### 9.3 iOS Safari Notes

iOS Safari is stricter with self-signed certificates. If the above bypass doesn't work:

1. Open `https://192.168.1.100:3443/certs/server.crt` in Safari on the phone (if you configured a cert download path), download and install the certificate
2. Or send `server.crt` to the phone via AirDrop / WeChat, then go to **Settings → General → Profile → Install**
3. Then go to **Settings → General → About → Certificate Trust Settings → Enable Full Trust**

After that, Safari can access `https://192.168.1.100:3443` normally.

---

## 10. Common Issues and Troubleshooting

### 10.1 Frontend Opens, but Cannot Start a Conversation

Check backend logs:

```bash
docker logs --tail 200 minicpmo-backend
```

Key things to look for:

- Whether the model path exists: `/models/MiniCPM-o-4_5`
- Whether VRAM is sufficient (H100 usually has enough)
- Whether `trust_remote_code` or dependency version errors appear

### 10.2 GPU Not Visible Inside Container

```bash
docker exec -it minicpmo-backend nvidia-smi
```

If it fails, check the NVIDIA Container Toolkit and Docker daemon configuration first.

### 10.3 WebSocket / SSE Anomalies

This project has already disabled buffering and configured WebSocket upgrade in `nginx.docker.conf`.
If issues persist, check whether the company's intranet gateway is blocking long-lived connections.

### 10.4 Model Startup Is Too Slow

The first startup may be slow; subsequent starts will be much faster. Check with:

```bash
nvidia-smi
docker logs -f minicpmo-backend
```

---

## 11. Optional Optimizations for Next Steps

1. Switch the backend image to "offline wheel installation mode" to completely eliminate the need for pip internet access on the server.
2. Use a private image registry (Harbor) instead of tar package transfers.
3. Use systemd or cron for automatic container restart and log rotation.
4. Replace the self-signed certificate with one issued by an enterprise CA to eliminate manual trust on mobile devices.

---

## 12. One-Click Command Quick Reference

### H100 Side (Assuming Files Are Already Uploaded)

```bash
cd /data/minicpmo/deploy_pkg

docker load -i minicpmo-backend_latest.tar
docker load -i minicpmo-frontend_latest.tar

mkdir -p /data/minicpmo/runtime/certs
cp docker-compose.yml /data/minicpmo/runtime/
cp certs/server.* /data/minicpmo/runtime/certs/

cd /data/minicpmo/runtime
export MODEL_PATH=/data/models/MiniCPM-o-4_5
export CERTS_PATH=./certs
export BACKEND_PORT=32550
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "Compose not found. Please install docker-compose or the docker compose plugin first." && exit 1
fi

$COMPOSE_CMD -f docker-compose.yml up -d
```

### Local Computer (Open Tunnel)

```bash
ssh -N -p $SSH_PORT -L 3000:127.0.0.1:3000 -L 3443:127.0.0.1:3443 -L 32550:127.0.0.1:32550 $SSH_USER@$SSH_HOST
```

PowerShell version:

```powershell
ssh -N -p $env:SSH_PORT -L 3000:127.0.0.1:3000 -L 3443:127.0.0.1:3443 -L 32550:127.0.0.1:32550 "$env:SSH_USER@$env:SSH_HOST"
```

Open on local computer: <http://127.0.0.1:3000>

### Mobile (Relayed Through Laptop)

```bash
# Bind all interfaces on the laptop
ssh -N -p $SSH_PORT -L 0.0.0.0:3443:127.0.0.1:3443 $SSH_USER@$SSH_HOST
```

Open on mobile browser: `https://<laptop_lan_ip>:3443`
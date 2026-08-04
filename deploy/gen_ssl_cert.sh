#!/bin/bash
# ============================================
# Generate self-signed SSL certificate (for Nginx HTTPS + mobile access)
# Usage: bash deploy/gen_ssl_cert.sh [output directory]
# ============================================
set -e

OUT_DIR="${1:-<YOUR_CERTS_OUTPUT_DIR>}"
mkdir -p "$OUT_DIR"

echo ">>> Generating self-signed SSL certificate to $OUT_DIR ..."
openssl req -x509 -nodes -days 3650 \
    -newkey rsa:2048 \
    -keyout "$OUT_DIR/server.key" \
    -out "$OUT_DIR/server.crt" \
    -subj "/C=CN/ST=Local/L=Local/O=MiniCPMo/OU=Dev/CN=<YOUR_CN>" \
    -addext "subjectAltName=IP:<YOUR_IP1>,IP:<YOUR_IP2>,DNS:<YOUR_DNS>"

echo ">>> Certificate generated:"
ls -lh "$OUT_DIR"/server.*
echo ""
echo ">>> Tip: After uploading the entire $OUT_DIR directory to the server,"
echo "    create a certs/ directory next to docker-compose.yml and put server.crt + server.key inside"

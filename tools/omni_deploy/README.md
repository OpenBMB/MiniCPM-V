# MiniCPM-o Omni One-Click Bootstrap

This folder provides a single-file bootstrap script for MiniCPM-o omni docker workflow.

## Files

- `bootstrap_minicpm_omni.sh`: main one-click bootstrap script

## Quick Start (Linux/macOS)

```bash
cd tools/omni_deploy
chmod +x bootstrap_minicpm_omni.sh
./bootstrap_minicpm_omni.sh --source cn --ask-sudo
```

## Notes

- `--source` supports `cn` and `global`.
- Script prefers downloading files to current workspace.
- sudo password is local-only, memory-only, never written to disk.
- If `omni_docker/` is missing, script can download `omni_docker.zip` by URL or Google Drive file-id.

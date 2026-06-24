# API Guide

This document introduces how to use the APIs for MiniCPM-V 4.6 and MiniCPM-o 4.5.

## MiniCPM-V 4.6

MiniCPM-V 4.6 can be called through the Chat Completions API. The interface supports both text-only and vision-language requests.

### Endpoint

```text
Base URL: https://api.modelbest.co/v1
Chat API: POST /chat/completions
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

A **free** public API key is currently available for trying the service:

```text
lis_sk_298cf78155f231c7_DkrDcNLHnK8dJRnfFrJCd4JGDbBLMkHrC3T-wLpvC9zy0BPemsyFuQ
```

Available model IDs:

```text
MiniCPM-V-4.6-Instruct
MiniCPM-V-4.6-Thinking
```

### Text-Only Request

```bash
curl https://api.modelbest.co/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.6-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Introduce yourself in one sentence."
      }
    ]
  }'
```

To use the Thinking model, replace the `model` value with:

```json
"MiniCPM-V-4.6-Thinking"
```

### Vision-Language Request

Images are passed as base64 data URLs in the `image_url` content format.

```bash
curl https://api.modelbest.co/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.6-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,<BASE64_IMAGE>"
            }
          }
        ]
      }
    ]
  }'
```

### Python Example

```python
import json
import urllib.request

api_key = "<API_KEY>"
payload = {
    "model": "MiniCPM-V-4.6-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "List three use cases for MiniCPM-V.",
        }
    ],
}

request = urllib.request.Request(
    "https://api.modelbest.co/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

with urllib.request.urlopen(request) as response:
    data = json.loads(response.read().decode("utf-8"))

print(data["choices"][0]["message"]["content"])
```

## MiniCPM-o 4.5

MiniCPM-o 4.5 can be called through the Chat Completions API for both traditional text and vision-language requests, and also full-duplex realtime interaction.

### Text-Only Request

```bash
curl https://api.modelbest.co/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-o-4.5",
    "messages": [
      {
        "role": "user",
        "content": "Introduce yourself in one sentence."
      }
    ]
  }'
```

### Vision-Language Request

```bash
curl https://api.modelbest.co/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-o-4.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,<BASE64_IMAGE>"
            }
          }
        ]
      }
    ]
  }'
```

### Realtime API

MiniCPM-o 4.5 also supports full-duplex realtime multimodal interaction. For the Realtime API documentation, see the [Realtime API Overview](https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/).

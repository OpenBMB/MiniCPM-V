# API Guide

This document introduces how to use the Chat Completions APIs for MiniCPM-V 4.5 / 4.6 / 4.7 and MiniCPM-o 4.5, including text, image, and video understanding.

## Endpoint

```text
Base URL: https://api.modelbest.cn/v1
Chat API: POST /chat/completions
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

The API is OpenAI-compatible. Create an API key at [platform.modelbest.cn](https://platform.modelbest.cn), then:

```bash
export API_KEY="your-api-key"
```

## Available Models

```text
MiniCPM-V-4.7-1B
MiniCPM-V-4.7-35B-A3B
MiniCPM-V-4.5-9B
MiniCPM-V-4.6-1B
MiniCPM-V-4.6-Thinking
MiniCPM-O-4.5-9B
```

## MiniCPM-V 4.7

MiniCPM-V 4.7 comes in two sizes — `MiniCPM-V-4.7-1B` (dense, edge-friendly) and `MiniCPM-V-4.7-35B-A3B` (MoE, high-throughput). Both support text-only, image, and video understanding through the Chat Completions API. Pick the size by setting the `model` field.

Thinking is **off by default**; enable it per request with `chat_template_kwargs` (see below). The reasoning trace is returned in `message.reasoning`.

### Text-Only Request

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.7-1B",
    "messages": [
      {
        "role": "user",
        "content": "Introduce yourself in one sentence."
      }
    ]
  }'
```

For the MoE size, replace the `model` value with `"MiniCPM-V-4.7-35B-A3B"`.

To enable thinking, add:

```json
"chat_template_kwargs": {"enable_thinking": true}
```

The thinking model returns its intermediate reasoning in `message.reasoning` and the final answer in `message.content`.

### Image Understanding

Images are passed as base64 data URLs in the `image_url` content format.

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.7-1B",
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

### Video Understanding

Videos are passed as base64 data URLs in the `video_url` content format (for example `data:video/mp4;base64,...`).

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.7-1B",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this video briefly."
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "data:video/mp4;base64,<BASE64_VIDEO>"
            }
          }
        ]
      }
    ]
  }'
```

## MiniCPM-V 4.6

MiniCPM-V 4.6 can be called through the Chat Completions API. The interface supports text-only, image, and video understanding requests.

### Text-Only Request

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.6-1B",
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

The Thinking model may return intermediate reasoning in `message.reasoning`.

### Image Understanding

Images are passed as base64 data URLs in the `image_url` content format.

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.6-1B",
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

### Video Understanding

Videos are passed as base64 data URLs in the `video_url` content format (for example `data:video/mp4;base64,...`). The same format works for `MiniCPM-V-4.7-1B`, `MiniCPM-V-4.7-35B-A3B`, `MiniCPM-V-4.5-9B`, `MiniCPM-V-4.6-1B`, `MiniCPM-V-4.6-Thinking`, and `MiniCPM-O-4.5-9B`.

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.6-1B",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this video briefly."
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "data:video/mp4;base64,<BASE64_VIDEO>"
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
import os
import urllib.request

api_key = os.environ["API_KEY"]
payload = {
    "model": "MiniCPM-V-4.6-1B",
    "messages": [
        {
            "role": "user",
            "content": "List three use cases for MiniCPM-V.",
        }
    ],
}

request = urllib.request.Request(
    "https://api.modelbest.cn/v1/chat/completions",
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

Or with the OpenAI SDK:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.modelbest.cn/v1",
)

resp = client.chat.completions.create(
    model="MiniCPM-V-4.6-1B",
    messages=[{"role": "user", "content": "List three use cases for MiniCPM-V."}],
)
print(resp.choices[0].message.content)
```

## MiniCPM-V 4.5

Same Chat Completions API as above (text / image / video). Use model ID `MiniCPM-V-4.5-9B`:

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.5-9B",
    "messages": [
      {
        "role": "user",
        "content": "Introduce yourself in one sentence."
      }
    ]
  }'
```

## MiniCPM-o 4.5

MiniCPM-o 4.5 can be called through the Chat Completions API for text, image, and video understanding requests, and also full-duplex realtime interaction.

### Text-Only Request

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-O-4.5-9B",
    "messages": [
      {
        "role": "user",
        "content": "Introduce yourself in one sentence."
      }
    ]
  }'
```

### Image Understanding

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-O-4.5-9B",
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

### Video Understanding

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-O-4.5-9B",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this video briefly."
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "data:video/mp4;base64,<BASE64_VIDEO>"
            }
          }
        ]
      }
    ]
  }'
```

### Realtime API

MiniCPM-o 4.5 also supports full-duplex realtime multimodal interaction. For the Realtime API documentation, see the [Realtime API Overview](https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/).

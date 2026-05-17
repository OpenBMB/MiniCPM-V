# API Guide

This document introduces how to use the APIs for MiniCPM-V 4.6 and MiniCPM-o 4.5.

## MiniCPM-V 4.6

MiniCPM-V 4.6 can be called through the Chat Completions API. The interface supports both text-only and vision-language requests.

### Endpoint

```text
Base URL: https://api.modelbest.cn/v1
Chat API: POST /chat/completions
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

A **free** public API key is currently available for trying the service:

```text
sk-pQ8L2zF3XmR5kY9wV4jB7hN1tC6vM0xG3aD5sH2bJ9lK4cZ8
```

Available model IDs:

```text
MiniCPM-V-4.6-1.3B-Instruct
MiniCPM-V-4.6-1.3B-think-0506_tau2_rl
```

### Text-Only Request

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.6-1.3B-Instruct",
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
"MiniCPM-V-4.6-1.3B-think-0506_tau2_rl"
```

### Vision-Language Request

Images are passed as base64 data URLs in the `image_url` content format.

```bash
curl https://api.modelbest.cn/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-4.6-1.3B-Instruct",
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
    "model": "MiniCPM-V-4.6-1.3B-Instruct",
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

## MiniCPM-o 4.5

MiniCPM-o 4.5 provides a Realtime API for full-duplex multimodal interaction. For the current MiniCPM-o 4.5 API documentation, see the [Realtime API Overview](https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/).

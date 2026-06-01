# Project Instructions

## Shell

- Follow `/home/sbplab/frank/.tools/codex/config/RTK.md`.
- Prefix shell commands with `rtk`.

## Image Generation

- When the user asks to generate a bitmap image for this project, default to the InROI image API command below unless the user explicitly asks for another workflow.
- Save generated images under `output/imagegen/` unless the user specifies a different path.

```bash
rtk proxy bash -lc 'curl https://www.inroi.shop/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INROI_API_KEY:?set INROI_API_KEY}" \
  -d '"'"'{
    "model": "gpt-image-2",
    "prompt": "A simple blue apple on a white background",
    "size": "1024x1024",
    "quality": "medium"
  }'"'"' \
  | jq -r ".data[0].b64_json" \
  | base64 -d > output/imagegen/output.png'
```

- Replace `prompt`, `size`, `quality`, and the output filename as needed.

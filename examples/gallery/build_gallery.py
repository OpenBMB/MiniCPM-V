import argparse
import html
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a static MiniCPM-V examples gallery from JSON outputs.")
    parser.add_argument(
        "--result-json",
        action="append",
        required=True,
        type=Path,
        help="Path to a JSON file written by minicpmv46_inference.py. Pass multiple times.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/minicpmv46_gallery/index.html"))
    parser.add_argument("--title", default="MiniCPM-V 4.6 Gallery")
    return parser.parse_args()


def to_display_src(value, output_path):
    if not value:
        return ""
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value)
    if not path.exists():
        return value
    try:
        return Path(os.path.relpath(path.resolve(), output_path.resolve().parent)).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def load_case(path, output_path):
    data = json.loads(path.read_text(encoding="utf-8"))
    media = data.get("media")
    if not media:
        media = []
        for image_url in data.get("image_url") or []:
            media.append({"type": "image", "url": image_url})
        if data.get("video_url"):
            media.append({"type": "video", "url": data["video_url"]})

    title = path.stem.replace("_", " ").title()
    normalized_media = []
    for item in media:
        normalized_media.append(
            {
                "type": item.get("type", "image"),
                "src": to_display_src(item.get("url", ""), output_path),
            }
        )

    return {
        "title": title,
        "prompt": data.get("prompt", ""),
        "answer": data.get("answer", ""),
        "media": normalized_media,
        "source": path.as_posix(),
    }


def write_gallery(cases, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases_json = json.dumps(cases, ensure_ascii=False)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #f6f7fb; color: #172033; font-family: Arial, sans-serif; }}
main {{ max-width: 1120px; margin: 28px auto; padding: 0 18px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
.subtitle {{ margin: 0 0 22px; color: #475569; }}
.card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 18px; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06); }}
.top {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 16px; }}
.case-title {{ margin: 0; font-size: 22px; }}
.buttons {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; max-width: 680px; }}
button {{ border: 1px solid #cbd5e1; border-radius: 6px; background: white; padding: 8px 12px; cursor: pointer; font-size: 14px; }}
button.active {{ background: #172033; color: white; border-color: #172033; }}
.media {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
.frame {{ background: #eef2f7; border: 1px solid #d8dee9; border-radius: 6px; padding: 10px; }}
.frame img, .frame video {{ width: 100%; max-height: 560px; object-fit: contain; background: white; border-radius: 5px; }}
.label {{ margin: 0 0 8px; color: #475569; font-weight: 700; }}
.section-title {{ margin: 18px 0 8px; color: #475569; font-size: 15px; text-transform: uppercase; letter-spacing: 0.04em; }}
pre {{ white-space: pre-wrap; background: #f1f5f9; border-radius: 6px; padding: 12px; margin: 0; overflow: auto; }}
@media (max-width: 760px) {{ .top {{ display: block; }} .buttons {{ justify-content: flex-start; margin-top: 12px; }} }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="subtitle">Local smoke-test inputs and model outputs generated with <code>examples/gallery/minicpmv46_inference.py</code>.</p>
<section class="card">
  <div class="top">
    <h2 class="case-title" id="caseTitle"></h2>
    <div class="buttons" id="caseButtons"></div>
  </div>
  <div class="media" id="media"></div>
  <h3 class="section-title">Prompt</h3>
  <pre id="prompt"></pre>
  <h3 class="section-title">Model Output</h3>
  <pre id="answer"></pre>
</section>
</main>
<script>
const cases = {cases_json};
const buttons = document.getElementById('caseButtons');
const media = document.getElementById('media');
function renderCase(index) {{
  const item = cases[index];
  document.getElementById('caseTitle').textContent = item.title;
  document.getElementById('prompt').textContent = item.prompt;
  document.getElementById('answer').textContent = item.answer;
  media.innerHTML = '';
  item.media.forEach((entry, mediaIndex) => {{
    const frame = document.createElement('section');
    frame.className = 'frame';
    const label = document.createElement('p');
    label.className = 'label';
    label.textContent = entry.type === 'video' ? 'Video' : `Image ${{mediaIndex + 1}}`;
    frame.appendChild(label);
    const element = document.createElement(entry.type === 'video' ? 'video' : 'img');
    element.src = entry.src;
    if (entry.type === 'video') {{
      element.controls = true;
      element.muted = true;
    }} else {{
      element.alt = item.title;
    }}
    frame.appendChild(element);
    media.appendChild(frame);
  }});
  [...buttons.children].forEach((button, buttonIndex) => button.classList.toggle('active', buttonIndex === index));
}}
cases.forEach((item, index) => {{
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = item.title;
  button.addEventListener('click', () => renderCase(index));
  buttons.appendChild(button);
}});
if (cases.length) {{
  renderCase(0);
}}
</script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


def main():
    args = parse_args()
    cases = [load_case(path, args.output) for path in args.result_json]
    write_gallery(cases, args.output, args.title)
    print(f"Wrote gallery: {args.output}")


if __name__ == "__main__":
    main()

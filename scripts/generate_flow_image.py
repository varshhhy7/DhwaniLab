import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "project_flow_prompt.txt"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEGATIVE = (
    "blurry, low resolution, jpeg artifacts, watermark, signature, glare, "
    "reflection, hands, fingers, people, person, 3d render, oil painting, "
    "colourful background, dark background, perspective distortion, cropped edges"
)


def load_prompt(path, section):
    text = Path(path).read_text(encoding="utf-8")
    marker = "PROMPT {} ".format(section.upper())
    if marker not in text:
        return text.strip()
    body = text.split(marker, 1)[1]
    body = body.split("=" * 80, 1)[1] if "=" * 80 in body else body
    body = body.split("=" * 80, 1)[0]
    return body.strip()


def save(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print("wrote", out_path, "({:,} bytes)".format(len(data)))


def run_local(prompt, args):
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        sys.exit(
            "local backend needs: pip install torch diffusers transformers accelerate"
        )

    cuda = torch.cuda.is_available()
    print("device:", "cuda" if cuda else "cpu")
    if cuda:
        print("gpu   :", torch.cuda.get_device_name(0))

    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if cuda else torch.float32,
        variant="fp16" if cuda else None,
    )
    pipe = pipe.to("cuda" if cuda else "cpu")
    pipe.enable_attention_slicing()
    if cuda:
        pipe.enable_model_cpu_offload()

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": args.steps,
        "width": args.width,
        "height": args.height,
    }
    if "turbo" not in args.model:
        kwargs["negative_prompt"] = NEGATIVE
        kwargs["guidance_scale"] = args.guidance
    else:
        kwargs["guidance_scale"] = 0.0

    generator = None
    if args.seed is not None:
        generator = torch.Generator("cuda" if cuda else "cpu").manual_seed(args.seed)
        kwargs["generator"] = generator

    image = pipe(**kwargs).images[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print("wrote", out_path)


def _post(url, payload, headers, timeout=300):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_openai(prompt, args):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY")

    payload = {
        "model": args.model if args.model != DEFAULTS["local"] else "gpt-image-1",
        "prompt": prompt,
        "size": "{}x{}".format(args.width, args.height),
        "n": 1,
    }
    data = _post(
        "https://api.openai.com/v1/images/generations",
        payload,
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    entry = data["data"][0]
    if entry.get("b64_json"):
        save(base64.b64decode(entry["b64_json"]), args.out)
    else:
        with urllib.request.urlopen(entry["url"], timeout=300) as response:
            save(response.read(), args.out)


def run_stability(prompt, args):
    key = os.environ.get("STABILITY_API_KEY")
    if not key:
        sys.exit("set STABILITY_API_KEY")

    payload = {
        "prompt": prompt,
        "output_format": "png",
        "aspect_ratio": "16:9",
        "negative_prompt": NEGATIVE,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    data = _post(
        "https://api.stability.ai/v2beta/stable-image/generate/core",
        payload,
        {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    save(base64.b64decode(data["image"]), args.out)


DEFAULTS = {
    "local": "stabilityai/sdxl-turbo",
    "openai": "gpt-image-1",
    "stability": "core",
}

BACKENDS = {
    "local": run_local,
    "openai": run_openai,
    "stability": run_stability,
}


def autodetect():
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("STABILITY_API_KEY"):
        return "stability"
    return "local"


def main():
    parser = argparse.ArgumentParser(
        description="Generate the project flow diagram as an image."
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--section", default="b", choices=["a", "b"],
                        help="a = full detail, b = simplified (better for SD)")
    parser.add_argument("--prompt", default=None, help="override the prompt text")
    parser.add_argument("--out", default="reports/project_flow.png")
    parser.add_argument("--model", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    backend = args.backend or autodetect()
    if args.model is None:
        args.model = DEFAULTS[backend]

    prompt = args.prompt or load_prompt(args.prompt_file, args.section)

    print("backend:", backend)
    print("model  :", args.model)
    if args.show_prompt:
        print()
        print(prompt)
        print()

    BACKENDS[backend](prompt, args)
    print()
    print("Image models garble small text. For a diagram with correct labels,")
    print("use scripts/render_flow_diagram.py instead.")


if __name__ == "__main__":
    main()

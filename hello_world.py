#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path
from typing import List

from PIL import Image
from transformers import pipeline


def find_images(paths: List[str]) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
    results: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in exts:
                results.extend(Path(path).rglob(f"*{ext}"))
        else:
            # Support globs and single files
            for m in glob.glob(p):
                mp = Path(m)
                if mp.is_file() and mp.suffix.lower() in exts:
                    results.append(mp)
    # De-dupe and sort for stable order
    uniq = sorted({x.resolve() for x in results})
    return uniq


def main():
    parser = argparse.ArgumentParser(description="Image caption generator")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files, globs, or directories to caption",
    )
    parser.add_argument(
        "--model",
        default="nlpconnect/vit-gpt2-image-captioning",
        help="Hugging Face model id (defaults to a lightweight captioner)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate for each caption",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Beam search width (higher can improve quality but is slower)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print captions (omit file paths)",
    )

    args = parser.parse_args()

    images = find_images(args.inputs)
    if not images:
        print("No images found.")
        return

    # Create pipeline (device auto-selection handled by HF/torch)
    pipe = pipeline(
        task="image-to-text",
        model=args.model,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
    }

    # Process in mini-batches
    for i in range(0, len(images), args.batch_size):
        batch_paths = images[i : i + args.batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        outputs = pipe(batch_imgs, generate_kwargs=gen_kwargs)
        # outputs is a list of lists of dicts or list of dicts depending on version
        # Normalize to list[dict]
        if outputs and isinstance(outputs[0], list):
            outputs = [o[0] for o in outputs]
        for p, o in zip(batch_paths, outputs):
            caption = o.get("generated_text") or o.get("caption") or ""
            if args.quiet:
                print(caption)
            else:
                print(f"{p}: {caption}")


if __name__ == "__main__":
    main()

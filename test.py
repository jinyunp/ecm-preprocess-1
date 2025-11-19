#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys

def main(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    paragraphs = doc.get("paragraphs", [])
    for i, p in enumerate(paragraphs):
        style = p.get("style")
        text = (p.get("text") or "").strip()
        print(f"[{i:04d}] style={style!r}  text_preview={text[:40]!r}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_styles.py <sanitized_json_path>")
        sys.exit(1)
    main(sys.argv[1])
#!/usr/bin/env python3
"""
SRT AI Translator - Translate SRT subtitle files using OpenAI API
"""

import argparse
import sys
from pathlib import Path
import pysrt
from openai import OpenAI
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re


def main():
    parser = argparse.ArgumentParser(description="Translate SRT subtitle files using OpenAI API")
    
    parser.add_argument("srt_file", help="Path to the input SRT file")
    parser.add_argument("base_url", help="Base URL for the OpenAI API")
    parser.add_argument("model", help="Model name to use for translation")
    parser.add_argument("output_path", help="Path for the output translated SRT file")
    
    parser.add_argument("--window-size", type=int, default=4, 
                       help="Number of subtitle entries to process in each batch (default: 4)")
    parser.add_argument("--srt-context", default="", 
                       help="Context information for translation (e.g., video type, dialect)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.srt_file).exists():
        print(f"Error: SRT file '{args.srt_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing: {args.srt_file}")
    print(f"Window size: {args.window_size}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_path}")
    if args.srt_context:
        print(f"Context: {args.srt_context}")
    
    # Parse SRT file
    try:
        subs = pysrt.open(args.srt_file)
        print(f"Loaded {len(subs)} subtitle entries")
    except Exception as e:
        print(f"Error parsing SRT file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

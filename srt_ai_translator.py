#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pysrt",
#   "openai",
#   "tqdm",
# ]
# ///
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
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitle files using OpenAI API"
    )

    parser.add_argument("srt_file", help="Path to the input SRT file")
    parser.add_argument("base_url", help="Base URL for the OpenAI API")
    parser.add_argument("model", help="Model name to use for translation")
    parser.add_argument("output_path", help="Path for the output translated SRT file")

    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Number of subtitle entries to process in each batch (default: 4)",
    )
    parser.add_argument(
        "--srt-context",
        default="",
        help="Context information for translation (e.g., video type, dialect)",
    )

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

    # Initialize OpenAI client
    client = OpenAI(base_url=args.base_url)

    # Process subtitles in windows
    translated_subs = pysrt.SubRipFile()

    # Create windows of subtitles
    windows = []
    for i in range(0, len(subs), args.window_size):
        window = subs[i : i + args.window_size]
        windows.append(window)

    print(f"Processing {len(windows)} windows...")

    # Process each window with progress bar
    for window_idx, window in enumerate(tqdm(windows, desc="Translating")):
        translated_window = translate_window(
            client, window, args.model, args.srt_context
        )
        translated_subs.extend(translated_window)

    # Save translated SRT
    try:
        translated_subs.save(args.output_path, encoding="utf-8")
        print(f"Translation saved to: {args.output_path}")
    except Exception as e:
        print(f"Error saving translated SRT: {e}", file=sys.stderr)
        sys.exit(1)


def translate_window(client, window, model, context):
    """Translate a window of subtitle entries using OpenAI API"""
    # Build XML prompt
    xml_texts = []
    for i, sub in enumerate(window, 1):
        xml_texts.append(f'<text id="{i}">{sub.text}</text>')

    context_xml = (
        f"<srt-context>{context}</srt-context>"
        if context
        else "<srt-context></srt-context>"
    )

    prompt = f"""Please translate the following subtitle texts to English. Think about the context and provide accurate translations.

{context_xml}

{chr(10).join(xml_texts)}

Please think about the translation, then provide your answer in this exact format:
<answer>
<text id="1">translated text here</text>
<text id="2">translated text here</text>
...
</answer>"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            response_text = response.choices[0].message.content
            translated_texts = parse_xml_response(response_text, len(window))

            # Create translated subtitle entries
            translated_window = []
            for i, (original_sub, translated_text) in enumerate(
                zip(window, translated_texts)
            ):
                new_sub = pysrt.SubRipItem(
                    index=original_sub.index,
                    start=original_sub.start,
                    end=original_sub.end,
                    text=translated_text,
                )
                translated_window.append(new_sub)

            return translated_window

        except Exception as e:
            if attempt < max_retries - 1:
                # Retry with error message
                prompt += f"\n\nPrevious parsing failed with error: {str(e)}. Please correct the format and try again."
                continue
            else:
                print(
                    f"Failed to translate window after {max_retries} attempts: {e}",
                    file=sys.stderr,
                )
                # Return original texts as fallback
                return window


def parse_xml_response(response_text, expected_count):
    """Parse XML response and extract translated texts"""
    try:
        # Extract answer block
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if not answer_match:
            raise ValueError("No <answer> block found in response")

        answer_content = answer_match.group(1)

        # Parse individual text elements
        text_pattern = r'<text id="(\d+)">(.*?)</text>'
        matches = re.findall(text_pattern, answer_content, re.DOTALL)

        if len(matches) != expected_count:
            raise ValueError(
                f"Expected {expected_count} text elements, found {len(matches)}"
            )

        # Sort by ID and extract texts
        matches.sort(key=lambda x: int(x[0]))
        translated_texts = [match[1].strip() for match in matches]

        return translated_texts

    except Exception as e:
        raise ValueError(f"XML parsing failed: {str(e)}")


if __name__ == "__main__":
    main()

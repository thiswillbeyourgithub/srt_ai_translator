#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pysrt",
#   "openai",
#   "tqdm",
#   "loguru",
# ]
# ///
"""
SRT AI Translator - Translate SRT subtitle files using OpenAI API
"""

import argparse
import sys
from pathlib import Path
import pysrt
from openai import OpenAI, APIConnectionError, AuthenticationError, APIStatusError
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
from loguru import logger


def main():
    # Configure logger to write to ./logs.txt with rotation and retention
    logger.add("./logs.txt", rotation="10 MB", retention="7 days", level="INFO")

    parser = argparse.ArgumentParser(
        description="Translate SRT subtitle files using OpenAI API"
    )

    parser.add_argument("--srt-file", required=True, help="Path to the input SRT file")
    parser.add_argument("--base-url", required=True, help="Base URL for the OpenAI API")
    parser.add_argument(
        "--model", required=True, help="Model name to use for translation"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path for the output translated SRT file",
    )

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
    parser.add_argument(
        "--target-language",
        default="English",
        help="Target language for translation (default: English)",
    )

    args = parser.parse_args()

    # Validate base URL starts with http:// or https://
    if not (args.base_url.startswith("http://") or args.base_url.startswith("https://")):
        logger.error(f"Base URL must start with http:// or https://, got: {args.base_url}")
        sys.exit(1)

    # Validate input file exists
    if not Path(args.srt_file).exists():
        logger.error(f"SRT file '{args.srt_file}' not found")
        sys.exit(1)

    logger.info(f"Processing: {args.srt_file}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Target language: {args.target_language}")
    if args.srt_context:
        logger.info(f"Context: {args.srt_context}")

    # Parse SRT file
    try:
        subs = pysrt.open(path=args.srt_file)
        logger.info(f"Loaded {len(subs)} subtitle entries")
    except Exception as e:
        logger.error(f"Error parsing SRT file: {e}")
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

    logger.info(f"Processing {len(windows)} windows...")

    # Process each window with progress bar
    for window_idx, window in enumerate(tqdm(iterable=windows, desc="Translating")):
        translated_window = translate_window(
            client=client,
            window=window,
            model=args.model,
            context=args.srt_context,
            target_language=args.target_language,
        )
        translated_subs.extend(translated_window)

    # Save translated SRT
    try:
        translated_subs.save(path=args.output_path, encoding="utf-8")
        logger.info(f"Translation saved to: {args.output_path}")
    except Exception as e:
        logger.error(f"Error saving translated SRT: {e}")
        sys.exit(1)


def translate_window(client, window, model, context, target_language):
    """Translate a window of subtitle entries using OpenAI API

    Parameters
    ----------
    client : OpenAI
        OpenAI client instance
    window : list
        List of subtitle entries to translate
    model : str
        Model name to use for translation
    context : str
        Context information for translation
    target_language : str
        Target language for translation
    """
    # Build XML prompt
    xml_texts = []
    for i, sub in enumerate(window, start=1):
        xml_texts.append(f'<text id="{i}">{sub.text}</text>')

    context_xml = (
        f"<srt-context>{context}</srt-context>"
        if context
        else "<srt-context></srt-context>"
    )

    prompt = f"""Please translate the following subtitle texts to {target_language}. Think about the context and provide accurate translations.

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

        except APIConnectionError as e:
            # Connection errors should crash immediately with helpful trace
            logger.error(f"API connection error - check network and base URL: {e}")
            raise
        except AuthenticationError as e:
            # Authentication errors should crash immediately
            logger.error(f"API authentication failed - check API key: {e}")
            raise
        except APIStatusError as e:
            # API status errors (like invalid model) should crash immediately
            logger.error(f"API returned error status - check model name and API: {e}")
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                # Retry with error message for parsing errors
                prompt += f"\n\nPrevious parsing failed with error: {str(e)}. Please correct the format and try again."
                continue
            else:
                logger.error(
                    f"Failed to translate window after {max_retries} attempts: {e}"
                )
                # Return original texts as fallback
                return window


def parse_xml_response(response_text, expected_count):
    """Parse XML response and extract translated texts"""
    try:
        # Extract answer block
        answer_match = re.search(
            pattern=r"<answer>(.*?)</answer>", string=response_text, flags=re.DOTALL
        )
        if not answer_match:
            raise ValueError("No <answer> block found in response")

        answer_content = answer_match.group(1)

        # Parse individual text elements
        text_pattern = r'<text id="(\d+)">(.*?)</text>'
        matches = re.findall(
            pattern=text_pattern, string=answer_content, flags=re.DOTALL
        )

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

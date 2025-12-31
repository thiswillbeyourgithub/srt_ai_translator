#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pysrt",
#   "openai",
#   "tqdm",
#   "loguru",
#   "ffmpeg-python",
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
import ffmpeg
import json
import tempfile

VERSION: str = "0.1.1"


def main():
    # Configure logger to write to ./logs.txt with rotation and retention
    logger.add("./logs.txt", rotation="10 MB", retention="7 days", level="INFO")

    parser = argparse.ArgumentParser(
        description="Translate SRT subtitle files using OpenAI API"
    )

    parser.add_argument("--srt-file", help="Path to the input SRT file")
    parser.add_argument(
        "--video",
        help="Path to the input video file (will extract subtitles)",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path for the output translated SRT file",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        help="Target language for translation (default: English)",
    )
    parser.add_argument(
        "--srt-context",
        default="",
        help="Context information for translation (e.g., video type, dialect)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Number of subtitle entries to process in each batch (default: 4)",
    )
    parser.add_argument(
        "--model", required=True, help="Model name to use for translation"
    )
    parser.add_argument("--base-url", required=True, help="Base URL for the OpenAI API")
    parser.add_argument(
        "--api-key",
        help="API key for OpenAI (if not provided, uses OPENAI_API_KEY environment variable)",
    )

    args = parser.parse_args()

    # Validate that either --srt-file or --video is provided (but not both)
    if not args.srt_file and not args.video:
        logger.error("Either --srt-file or --video must be provided")
        sys.exit(1)
    if args.srt_file and args.video:
        logger.error("Cannot specify both --srt-file and --video")
        sys.exit(1)

    # Validate base URL starts with http:// or https://
    if not (
        args.base_url.startswith("http://") or args.base_url.startswith("https://")
    ):
        logger.error(
            f"Base URL must start with http:// or https://, got: {args.base_url}"
        )
        sys.exit(1)

    # Handle video input - extract subtitles to temporary SRT file
    temp_srt_file = None
    if args.video:
        if not Path(args.video).exists():
            logger.error(f"Video file '{args.video}' not found")
            sys.exit(1)

        logger.info(f"Extracting subtitles from video: {args.video}")
        subtitle_streams = list_subtitle_streams(video_path=args.video)

        if not subtitle_streams:
            logger.error("No subtitle streams found in video")
            sys.exit(1)

        # If multiple streams, prompt user for choice
        selected_stream = None
        if len(subtitle_streams) == 1:
            selected_stream = subtitle_streams[0]
            logger.info(
                f"Found 1 subtitle stream: {selected_stream['language']} ({selected_stream['codec_name']})"
            )
        else:
            logger.info(f"Found {len(subtitle_streams)} subtitle streams:")
            for idx, stream in enumerate(subtitle_streams):
                lang = stream.get("language", "unknown")
                codec = stream.get("codec_name", "unknown")
                title = stream.get("title", "")
                title_str = f" - {title}" if title else ""

                # Get preview text to help identify the stream
                preview = get_subtitle_preview(
                    video_path=args.video, stream_index=stream["index"]
                )
                preview_str = f" | Preview: {preview}" if preview else ""

                logger.info(f"  {idx + 1}. {lang} ({codec}){title_str}{preview_str}")

            while True:
                try:
                    choice = input(
                        "Enter the number of the subtitle stream to translate: "
                    )
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(subtitle_streams):
                        selected_stream = subtitle_streams[choice_idx]
                        break
                    else:
                        print(
                            f"Invalid choice. Please enter a number between 1 and {len(subtitle_streams)}"
                        )
                except (ValueError, KeyboardInterrupt):
                    logger.error("Invalid input or interrupted")
                    sys.exit(1)

        # Extract subtitle to temporary SRT file
        # Using NamedTemporaryFile with delete=False to keep the file for processing
        # We'll clean it up at the end
        temp_srt_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False
        )
        temp_srt_path = temp_srt_file.name
        temp_srt_file.close()

        try:
            extract_subtitle_stream(
                video_path=args.video,
                stream_index=selected_stream["index"],
                output_path=temp_srt_path,
            )
            logger.info(f"Extracted subtitle to temporary file: {temp_srt_path}")
            args.srt_file = temp_srt_path
        except Exception as e:
            logger.error(f"Failed to extract subtitle: {e}")
            Path(temp_srt_path).unlink(missing_ok=True)
            sys.exit(1)

    # Validate input file exists
    if not Path(args.srt_file).exists():
        logger.error(f"SRT file '{args.srt_file}' not found")
        sys.exit(1)

    # Crash if output file already exists to prevent accidental overwrites
    if Path(args.output_path).exists():
        logger.error(
            f"Output file '{args.output_path}' already exists. Please remove it first or choose a different output path."
        )
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
    # If api_key is not provided, OpenAI client will use OPENAI_API_KEY environment variable
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Process subtitles in windows
    translated_subs = pysrt.SubRipFile()
    tmp_output_path = Path(args.output_path).with_suffix(
        Path(args.output_path).suffix + ".tmp"
    )

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

        # Write current progress to temporary file after each window
        # This allows continuous incremental updates and recovery on failure
        try:
            translated_subs.save(path=str(tmp_output_path), encoding="utf-8")
        except Exception as e:
            logger.error(f"Error saving progress to temporary file: {e}")
            sys.exit(1)

    # Atomically rename temp file to final output
    # This ensures the final output is written atomically
    try:
        tmp_output_path.rename(args.output_path)
        logger.info(f"Translation saved to: {args.output_path}")
    except Exception as e:
        logger.error(f"Error finalizing output file: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary SRT file if it was created from video extraction
        if temp_srt_file:
            Path(temp_srt_path).unlink(missing_ok=True)
            logger.info("Cleaned up temporary subtitle file")


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
    # Build XML prompt with timing information
    # Including timing helps the LLM understand temporal context (e.g., rapid dialogue vs. long pauses)
    xml_texts = []
    for i, sub in enumerate(window, start=1):
        xml_texts.append(
            f'<text id="{i}" start="{sub.start}" end="{sub.end}">{sub.text}</text>'
        )

    context_xml = (
        f"<srt-context>{context}</srt-context>"
        if context
        else "<srt-context></srt-context>"
    )

    prompt = f"""Please translate the following subtitle texts to {target_language}. Think about the context and provide accurate translations.

The timing information (start/end) is provided to help you understand the temporal context - whether this is rapid dialogue or if significant time has passed between subtitles.

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


def list_subtitle_streams(video_path: str) -> list:
    """List all subtitle streams in the video file

    Parameters
    ----------
    video_path : str
        Path to the video file

    Returns
    -------
    list
        List of subtitle stream information dictionaries
    """
    try:
        probe = ffmpeg.probe(filename=video_path)
        subtitle_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "subtitle"
        ]
        return subtitle_streams
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg error while probing video: {e.stderr.decode()}")
        raise
    except Exception as e:
        logger.error(f"Error probing video file: {e}")
        raise


def get_subtitle_preview(video_path: str, stream_index: int) -> str:
    """Get a preview text from a subtitle stream

    Extracts the first text that appears after the 5th text containing at least 10 characters.
    This helps identify subtitle streams when metadata is unclear.

    Parameters
    ----------
    video_path : str
        Path to the video file
    stream_index : int
        Index of the subtitle stream

    Returns
    -------
    str
        Preview text, or empty string if extraction fails or not enough content
    """
    temp_preview_file = None
    try:
        # Create temp file for preview extraction
        temp_preview_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False
        )
        temp_preview_path = temp_preview_file.name
        temp_preview_file.close()

        # Extract subtitle stream
        extract_subtitle_stream(
            video_path=video_path,
            stream_index=stream_index,
            output_path=temp_preview_path,
        )

        # Parse and find the target text
        subs = pysrt.open(path=temp_preview_path)

        # Find the 5th text with at least 10 characters
        # Track all seen texts to ensure preview is unique
        count = 0
        target_index = None
        seen_texts = set()
        
        for idx, sub in enumerate(subs):
            text = sub.text.strip()
            seen_texts.add(text)
            if len(text) >= 10:
                count += 1
                if count == 5:
                    # Now find the first unique text after this one
                    target_index = idx + 1
                    break

        # Get the first unique text after the 5th qualifying text
        if target_index is not None:
            for idx in range(target_index, len(subs)):
                preview_text = subs[idx].text.strip()
                # Skip if we've seen this text before
                if preview_text not in seen_texts:
                    # Limit preview length to avoid clutter
                    if len(preview_text) > 80:
                        preview_text = preview_text[:77] + "..."
                    return preview_text
                # Add to seen_texts to track duplicates
                seen_texts.add(preview_text)

        return ""

    except Exception as e:
        logger.debug(f"Could not extract preview for stream {stream_index}: {e}")
        return ""
    finally:
        # Clean up temp file
        if temp_preview_file:
            Path(temp_preview_path).unlink(missing_ok=True)


def extract_subtitle_stream(video_path: str, stream_index: int, output_path: str):
    """Extract a subtitle stream from video to SRT file

    Parameters
    ----------
    video_path : str
        Path to the video file
    stream_index : int
        Index of the subtitle stream to extract
    output_path : str
        Path for the output SRT file
    """
    try:
        # Use ffmpeg to extract the subtitle stream
        # Map the specific subtitle stream and convert to SRT format
        (
            ffmpeg.input(filename=video_path)
            .output(filename=output_path, map=f"0:{stream_index}", format="srt")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg error while extracting subtitle: {e.stderr.decode()}")
        raise
    except Exception as e:
        logger.error(f"Error extracting subtitle stream: {e}")
        raise


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

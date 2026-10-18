#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pysrt",
#   "litellm",
#   "tqdm",
#   "loguru",
#   "ffmpeg-python",
# ]
# ///
"""
SRT AI Translator - Translate SRT subtitle files using any LiteLLM-supported
provider (OpenRouter by default), with windowed processing and context awareness.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import pysrt
import litellm
from tqdm import tqdm
import re
from loguru import logger
import ffmpeg
import tempfile

VERSION: str = "1.0.0"

# Accepted finish_reason values from the completion API. Anything else means the
# provider did not cleanly finish the reply: "length" = truncated at the max
# token limit, "content_filter" = censored. Such a reply must be rejected and
# retried rather than parsed into a half-written (and likely malformed) result.
# None is accepted because some providers omit finish_reason on a normal reply.
_OK_FINISH_REASONS = {"stop", "end_turn", "completed", None}


class _HelpOnErrorParser(argparse.ArgumentParser):
    """ArgumentParser that prints the full help (with defaults) on a bad
    argument instead of only the terse usage line."""

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\n{self.prog}: error: {message}\n")


def main():
    # Configure logger to write to ./logs.txt with rotation and retention
    logger.add("./logs.txt", rotation="10 MB", retention="7 days", level="INFO")

    parser = _HelpOnErrorParser(
        description="Translate SRT subtitle files using LiteLLM (OpenRouter by default)"
    )

    parser.add_argument(
        "--input",
        required=True,
        metavar="SRT_OR_VIDEO",
        help="Path to the input file: either an SRT subtitle file (.srt) or a "
        "video file to extract subtitles from (any other extension, e.g. .mkv, .mp4).",
    )
    parser.add_argument(
        "--output",
        help="Path for the output translated SRT file (default: auto-generated based on input and target language)",
    )
    parser.add_argument(
        "--target-language",
        default="English",
        help="Target language for translation (default: %(default)s)",
    )
    parser.add_argument(
        "--file-description",
        default="",
        help="Description/context of the file to aid translation (e.g., video type, dialect, domain)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Number of subtitle entries to process in each batch (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the model (default: unset, so the "
        "provider/model default is used)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable model reasoning/thinking (sends OpenRouter reasoning "
        "{'enabled': False}). Useful with reasoning models like DeepSeek to make "
        "translation faster and cheaper. Omit to leave the provider default.",
    )
    parser.add_argument(
        "--model",
        default="openrouter/deepseek/deepseek-v4-pro",
        help="LiteLLM model name to use for translation (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        help="Optional API base URL override (LiteLLM api_base). Only needed for "
        "custom/self-hosted endpoints; not required for OpenRouter.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the provider. If not provided, LiteLLM falls back to the "
        "provider's standard environment variable (e.g. OPENROUTER_API_KEY).",
    )

    args = parser.parse_args()

    # Validate the input file exists up front (covers both subtitle and video).
    if not Path(args.input).exists():
        logger.error(f"Input file '{args.input}' not found")
        sys.exit(1)

    # Validate base URL only when explicitly provided (it is optional now that
    # LiteLLM routes to the provider based on the model prefix).
    if args.base_url and not (
        args.base_url.startswith("http://") or args.base_url.startswith("https://")
    ):
        logger.error(
            f"Base URL must start with http:// or https://, got: {args.base_url}"
        )
        sys.exit(1)

    # Decide how to treat --input from its extension: a .srt file is used
    # directly, anything else is treated as a video whose subtitles are
    # extracted to a temporary SRT file first. srt_file holds the SRT path
    # actually fed to the translator regardless of which branch was taken.
    is_srt_input = Path(args.input).suffix.lower() == ".srt"

    temp_srt_file = None
    temp_srt_path = None
    if is_srt_input:
        srt_file = args.input
    else:
        logger.info(f"Extracting subtitles from video: {args.input}")
        subtitle_streams = list_subtitle_streams(video_path=args.input)

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
                    video_path=args.input, stream_index=stream["index"]
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
                video_path=args.input,
                stream_index=selected_stream["index"],
                output_path=temp_srt_path,
            )
            logger.info(f"Extracted subtitle to temporary file: {temp_srt_path}")
            srt_file = temp_srt_path

            # Generate default output path if not provided
            # Uses video filename, stream language, and target language for clarity
            if not args.output:
                video_stem = Path(args.input).stem
                stream_lang = selected_stream.get("language", "unknown")
                args.output = f"{video_stem}_{stream_lang}_{args.target_language}.srt"
                logger.info(f"Auto-generated output path: {args.output}")
        except Exception as e:
            logger.error(f"Failed to extract subtitle: {e}")
            Path(temp_srt_path).unlink(missing_ok=True)
            sys.exit(1)

    # Generate default output path for SRT input if not provided
    # Uses SRT filename and target language
    if not args.output:
        srt_stem = Path(srt_file).stem
        args.output = f"{srt_stem}_{args.target_language}.srt"
        logger.info(f"Auto-generated output path: {args.output}")

    # Crash if output file already exists to prevent accidental overwrites.
    # This check is placed AFTER default output path generation so that it also
    # guards auto-generated paths (not just user-supplied ones) and never runs
    # against an unset (None) --output, which would raise a TypeError.
    if Path(args.output).exists():
        logger.error(
            f"Output file '{args.output}' already exists. Please remove it first or choose a different output path."
        )
        sys.exit(1)

    logger.info(f"Processing: {srt_file}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Target language: {args.target_language}")
    if args.file_description:
        logger.info(f"File description: {args.file_description}")

    # Parse SRT file
    try:
        subs = pysrt.open(path=srt_file)
        logger.info(f"Loaded {len(subs)} subtitle entries")
    except Exception as e:
        logger.error(f"Error parsing SRT file: {e}")
        sys.exit(1)

    # LiteLLM is stateless (no client object): credentials and the optional base
    # URL are passed per-call. When api_key is None, LiteLLM falls back to the
    # provider's standard environment variable (e.g. OPENROUTER_API_KEY).

    # Process subtitles in windows
    translated_subs = pysrt.SubRipFile()

    # Continuous-progress file: sits right next to the final output with ".tmp"
    # appended, and is rewritten after every window so a crash/interrupt leaves a
    # usable partial translation to recover from. Reset it up front so we never
    # resume on top of a stale ".tmp" left by a previous (crashed) run.
    tmp_output_path = Path(args.output).with_suffix(Path(args.output).suffix + ".tmp")
    try:
        atomic_save_srt(subs=translated_subs, dest=tmp_output_path)
    except Exception as e:
        logger.error(f"Error initializing temporary output file: {e}")
        sys.exit(1)

    # Create windows of subtitles
    windows = []
    for i in range(0, len(subs), args.window_size):
        window = subs[i : i + args.window_size]
        windows.append(window)

    logger.info(f"Processing {len(windows)} windows...")

    # Build the system prompt ONCE and reuse the identical object for every
    # window. All run-constant instructions (target language, file description,
    # output format) live here so the leading portion of every request is
    # byte-identical, which lets providers reuse their KV/prompt cache across
    # windows instead of re-processing the instructions each time.
    system_prompt = build_system_prompt(
        target_language=args.target_language, file_description=args.file_description
    )

    # Process each window with progress bar
    for window_idx, window in enumerate(tqdm(iterable=windows, desc="Translating")):
        translated_window = translate_window(
            window=window,
            model=args.model,
            system_prompt=system_prompt,
            api_key=args.api_key,
            api_base=args.base_url,
            temperature=args.temperature,
            no_thinking=args.no_thinking,
        )
        translated_subs.extend(translated_window)

        # Write current progress to the temporary file after each window for
        # continuous incremental updates and recovery on failure. The write is
        # atomic (see atomic_save_srt), so an interrupt mid-save never leaves a
        # corrupt ".tmp" - it keeps the previous window's complete contents.
        try:
            atomic_save_srt(subs=translated_subs, dest=tmp_output_path)
        except Exception as e:
            logger.error(f"Error saving progress to temporary file: {e}")
            sys.exit(1)

    # Atomically move the completed temp file onto the final output. os.replace
    # is atomic on the same filesystem (tmp_output_path is a sibling of output),
    # so the final file appears fully written or not at all.
    try:
        os.replace(tmp_output_path, args.output)
        logger.info(f"Translation saved to: {args.output}")
    except Exception as e:
        logger.error(f"Error finalizing output file: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary SRT file if it was created from video extraction
        if temp_srt_file:
            Path(temp_srt_path).unlink(missing_ok=True)
            logger.info("Cleaned up temporary subtitle file")


def atomic_save_srt(subs, dest) -> None:
    """Save a SubRipFile to dest atomically to avoid corruption on interruption.

    The subtitles are first written to a scratch file in the SAME directory as
    dest, then os.replace()'d onto dest. os.replace is atomic on a single
    filesystem (keeping the scratch a sibling of dest guarantees that), so a
    crash or interrupt during the write leaves dest either at its previous
    complete contents or fully updated, never half-written. The scratch file is
    removed if anything fails before the replace.

    Parameters
    ----------
    subs : pysrt.SubRipFile
        The subtitle file to persist.
    dest : str or pathlib.Path
        Destination path to atomically write.
    """
    dest = Path(dest)
    # ".part" scratch next to dest so the subsequent os.replace stays on one
    # filesystem (a cross-filesystem replace would not be atomic).
    scratch = dest.with_name(dest.name + ".part")
    try:
        subs.save(path=str(scratch), encoding="utf-8")
        os.replace(scratch, dest)
    finally:
        # Only reached with scratch still present if save() failed or os.replace
        # did not consume it; a successful replace leaves nothing to clean up.
        if scratch.exists():
            scratch.unlink(missing_ok=True)


def build_system_prompt(target_language: str, file_description: str) -> str:
    """Build the run-constant system prompt shared by every window.

    Everything that does not change between windows (target language, file
    description, output format and rules) lives here so that the leading portion
    of every request is byte-identical. Reusing an identical prefix is what lets
    providers serve it from their prompt/KV cache instead of re-processing it for
    each window, which is the whole point of factoring it out of the per-window
    user message.

    Parameters
    ----------
    target_language : str
        Target language for translation.
    file_description : str
        Description/context of the file to aid translation (may be empty).

    Returns
    -------
    str
        The system prompt.
    """
    file_description_xml = (
        f"<file-description>{file_description}</file-description>"
        if file_description
        else "<file-description></file-description>"
    )

    return f"""You are a professional subtitle translator. Translate every subtitle text the user sends into {target_language}, producing accurate and natural translations that respect the context.

Each subtitle is provided as an XML element of the form:
<text id="N" start="HH:MM:SS,mmm" end="HH:MM:SS,mmm">original text</text>
The start/end timing is context only (it tells you whether this is rapid dialogue or if significant time has passed between subtitles); never translate, echo or alter the ids or the timings.

The following describes the file being translated (may be empty):
{file_description_xml}

Reply with EXACTLY one <text> element per input subtitle, reusing the same ids, wrapped in a single <answer> block, and output nothing else, in this exact format:
<answer>
<text id="1">translated text here</text>
<text id="2">translated text here</text>
...
</answer>"""


def translate_window(
    window,
    model,
    system_prompt,
    api_key=None,
    api_base=None,
    temperature=None,
    no_thinking=False,
    max_retries=5,
    base_delay=1.0,
):
    """Translate a window of subtitle entries using a LiteLLM-supported provider

    Retries with exponential backoff. Three failure modes are retried: transient
    provider/network errors, replies the provider did not cleanly finish (bad
    finish_reason, e.g. truncation), and malformed replies (including replies that
    do not contain exactly one translated chunk per input subtitle, which is
    enforced by parse_xml_response). On a malformed reply the model's bad answer
    plus a correction are appended as extra turns, while the [system, user] prefix
    is kept byte-identical so it stays cacheable across retries and across windows.
    Authentication errors are never retried (they cannot succeed on retry).

    Parameters
    ----------
    window : list
        List of subtitle entries to translate.
    model : str
        LiteLLM model name to use for translation.
    system_prompt : str
        Run-constant system prompt (see build_system_prompt); identical across
        windows to maximize provider-side prompt/KV caching.
    api_key : str, optional
        API key for the provider. If None, LiteLLM uses the provider's standard
        environment variable.
    api_base : str, optional
        Optional API base URL override (LiteLLM api_base).
    temperature : float, optional
        Sampling temperature passed to the model. None (default) leaves it
        unset so the provider/model default applies.
    no_thinking : bool, optional
        When True, disable model reasoning/thinking by sending OpenRouter's
        reasoning {"enabled": False} in extra_body. When False, the reasoning
        field is omitted so the provider default applies (default: False).
    max_retries : int, optional
        Maximum number of attempts before falling back to the untranslated
        window (default: 5).
    base_delay : float, optional
        Base delay in seconds for exponential backoff; attempt N waits
        base_delay * 2**N seconds (default: 1.0).

    Returns
    -------
    list
        Translated SubRipItem entries, or the original (untranslated) window as a
        fallback if every attempt fails.
    """
    # Build the variable, per-window user message: only the subtitle texts (with
    # timing) go here. Keeping this separate from the constant system prompt is
    # what makes the cacheable prefix identical across windows.
    xml_texts = [
        f'<text id="{i}" start="{sub.start}" end="{sub.end}">{sub.text}</text>'
        for i, sub in enumerate(window, start=1)
    ]
    user_content = "Translate these subtitles:\n" + "\n".join(xml_texts)

    # The [system, user] prefix stays constant; correction turns are only ever
    # appended after it, preserving the cacheable prefix.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries):
        # --- Call the provider (transient errors are retried with backoff) ---
        try:
            completion_kwargs = dict(
                model=model,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
            )
            # Only send temperature when explicitly set, so the provider/model
            # default applies otherwise.
            if temperature is not None:
                completion_kwargs["temperature"] = temperature
            # Disabling reasoning ("thinking") is an OpenRouter request-body field
            # passed through extra_body. {"enabled": False} turns thinking off for
            # reasoning models like DeepSeek; omitting the field entirely leaves
            # the provider default (thinking on). Confirmed against OpenRouter.
            if no_thinking:
                completion_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

            response = litellm.completion(**completion_kwargs)
            choice = response.choices[0]

            # Finish-reason guard: reject a truncated ("length") or censored
            # ("content_filter") completion instead of parsing a half-written
            # reply. Raising here routes into the backoff-retry branch below.
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason not in _OK_FINISH_REASONS:
                raise RuntimeError(
                    f"provider ended the completion with finish_reason={finish_reason!r} "
                    f"('length' = truncated at the provider max, 'content_filter' = censored)"
                )

            response_text = choice.message.content
            if response_text is None:
                raise RuntimeError(
                    f"provider returned empty content (finish_reason={finish_reason!r})"
                )
        except litellm.AuthenticationError as e:
            # Authentication errors cannot be fixed by retrying, so crash.
            logger.error(f"API authentication failed - check API key: {e}")
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Provider call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            logger.error(
                f"Failed to translate window after {max_retries} attempts: {e}"
            )
            # Return original texts as fallback so the run can still complete.
            return window

        # --- Validate the reply: must parse and yield exactly len(window) chunks ---
        try:
            translated_texts = parse_xml_response(response_text, len(window))
        except Exception as e:
            # Log a truncated snippet of the raw reply so parsing failures are
            # diagnosable (e.g. spotting unexpected tag shapes) without dumping
            # the whole, potentially large, response into the logs.
            snippet = response_text.strip().replace("\n", " ")[:300]
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Invalid response (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s. Raw reply (truncated): {snippet!r}"
                )
                # Append the bad answer and a correction as new turns; the
                # [system, user] prefix above is left untouched for caching.
                messages = messages + [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            f"That response was invalid: {e}. Re-send the translation "
                            f"for ALL {len(window)} subtitles, exactly one <text> "
                            f"element per id, in the required <answer> format and nothing else."
                        ),
                    },
                ]
                time.sleep(delay)
                continue
            logger.error(
                f"Failed to parse a valid response after {max_retries} attempts: {e}. "
                f"Raw reply (truncated): {snippet!r}"
            )
            return window

        # --- Success: build the translated subtitle entries ---
        translated_window = []
        for original_sub, translated_text in zip(window, translated_texts):
            new_sub = pysrt.SubRipItem(
                index=original_sub.index,
                start=original_sub.start,
                end=original_sub.end,
                text=translated_text,
            )
            translated_window.append(new_sub)

        return translated_window


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
        # Extract answer block (case-insensitive so <Answer>/<ANSWER> also match)
        answer_match = re.search(
            pattern=r"<answer>(.*?)</answer>",
            string=response_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not answer_match:
            raise ValueError("No <answer> block found in response")

        answer_content = answer_match.group(1)

        # Parse individual <text> elements. The pattern is deliberately lenient
        # because models frequently mirror the input tag and re-emit the start/end
        # timing attributes (e.g. <text id="1" start="..." end="...">) despite the
        # instruction not to: [^>]* swallows any extra attributes after the id, and
        # the id quotes are optional. Being strict here caused every element to
        # fail to match ("Expected N text elements, found 0").
        text_pattern = r'<text\s+id=["\']?(\d+)["\']?[^>]*>(.*?)</text>'
        matches = re.findall(
            pattern=text_pattern,
            string=answer_content,
            flags=re.DOTALL | re.IGNORECASE,
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

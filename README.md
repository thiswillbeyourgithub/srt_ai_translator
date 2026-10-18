# SRT AI Translator

Translate SRT subtitle files using any [LiteLLM](https://github.com/BerriAI/litellm)-supported provider (OpenRouter by default) with intelligent windowed processing and context awareness.

## Features

- **Video Subtitle Extraction**: Extract subtitles directly from video files (pick a stream with `--subtitle-stream`)
- **Windowed Translation**: Processes subtitles in configurable batches to maintain context and optimize API usage
- **100+ Providers via LiteLLM**: Works with OpenRouter (default), OpenAI, Anthropic, local LLMs and any other LiteLLM-supported backend, selected purely by the model name
- **Previous-Window Context**: Each window is sent after the previous window(s) as past exchanges (original and translation), for continuity of split sentences, names and terminology (`--context-windows`)
- **Context-Aware**: Supports a custom file description for better translations (video type, dialect, domain-specific terminology)
- **Robust Error Handling**: Automatic retry with exponential backoff on transient provider errors, unclean completions (truncated/censored `finish_reason`), and malformed replies, with feedback to the model
- **Chunk-Count Validation**: Only accepts a reply that returns exactly one translated entry per input subtitle
- **Provider Pinning**: On OpenRouter, requests are pinned to DeepSeek's own endpoint by default with no fallback (`--provider`)
- **Reasoning Control**: Model reasoning is off by default for faster, cheaper runs; `--thinking`/`--reasoning` sets an effort level or leaves it to the provider
- **Prompt Cache Friendly**: Reuses an identical system prompt across every window to maximize provider-side prompt/KV caching
- **Progress Tracking**: Real-time progress bar using tqdm
- **Incremental Saves**: Saves progress after each window to allow recovery on failure
- **Type Hints & Docstrings**: Fully typed and documented codebase
- **Structured Prompts**: Uses XML-formatted prompts for precise parsing and control
- **Timing Context**: Includes subtitle timing information to help the model understand temporal context

## Installation

This script uses [PEP 723](https://peps.python.org/pep-0723/) inline script metadata, so you can run it directly with `uv`:

```bash
uv run srt_ai_translator.py --help
```

Or install dependencies manually:

```bash
pip install pysrt litellm tqdm loguru ffmpeg-python
```

**Note**: For video subtitle extraction, you also need `ffmpeg` installed on your system.

**API keys**: By default the tool uses OpenRouter, so set `OPENROUTER_API_KEY` (or pass `--api-key`). For other providers, use the model prefix LiteLLM expects (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4`) and set that provider's standard environment variable.

## Usage

### Basic Example (SRT File)

The default model is `openrouter/~deepseek/deepseek-pro-latest`, so with `OPENROUTER_API_KEY` set you only need the input:

```bash
export OPENROUTER_API_KEY="sk-or-..."
uv run srt_ai_translator.py \
  --input input.srt \
  --target-language English
```

The output path is auto-generated (here `input_English.srt`) unless you pass `--output`.

Reasoning (thinking) is disabled by default for a faster, cheaper run. To let the model think, set an effort level, or `default` to let OpenRouter/the model decide:

```bash
uv run srt_ai_translator.py \
  --input input.srt \
  --target-language English \
  --thinking high
```

### Extract and Translate from Video

Any `--input` that is not a `.srt` file is treated as a video and its subtitles are extracted first:

```bash
uv run srt_ai_translator.py \
  --input movie.mkv \
  --target-language English \
  --output movie_en.srt
```

If the video contains multiple subtitle streams and `--subtitle-stream` is not given, the tool lists them with their quoted name (the stream title, or its language when untitled), language, codec and a text preview, then exits:

```
  1. "fre" (fre, subrip)
  2. "English" (eng, subrip)
  3. "English SDH" (eng, subrip)
```

Rerun with `--subtitle-stream` set to either the 1-based number or the name. A name is matched exactly first, otherwise the unique name starting with it (case-insensitive) is used: `--subtitle-stream English` picks 2, `--subtitle-stream "english s"` picks 3, and `--subtitle-stream Eng` errors out because it is ambiguous (use the number then).

### With File Description and Custom Window Size

```bash
uv run srt_ai_translator.py \
  --input documentary.srt \
  --target-language English \
  --window-size 6 \
  --file-description "Scientific documentary about marine biology, use formal terminology"
```

### Using Another Provider or a Local LLM

Select the provider purely through the model name (and, for custom/self-hosted endpoints, `--base-url`):

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
uv run srt_ai_translator.py --input movie.srt --model openai/gpt-4o --target-language Spanish

# Local OpenAI-compatible server (e.g. LM Studio)
uv run srt_ai_translator.py \
  --input movie.srt \
  --model openai/local-model \
  --base-url http://localhost:1234/v1 \
  --api-key not-needed \
  --target-language Spanish
```

## Command-Line Arguments

### Required Arguments

- `--input`: Path to the input file: either an SRT subtitle file (`.srt`) or a video file to extract subtitles from (any other extension)

### Optional Arguments

- `--target-language`: Target language for translation (default: `English`)
- `--output`: Path for the output translated SRT file (if not provided, auto-generated based on input filename and target language)
- `--model`: LiteLLM model name to use for translation (default: `openrouter/~deepseek/deepseek-pro-latest`)
- `--base-url`: Optional API base URL override (LiteLLM `api_base`); only needed for custom/self-hosted endpoints
- `--api-key`: API key for the provider (if not provided, LiteLLM uses the provider's standard environment variable, e.g. `OPENROUTER_API_KEY`)
- `--window-size`: Number of subtitle entries to process in each batch (default: 8)
- `--file-description`: Description/context of the file to aid translation (e.g., video type, dialect, domain)
- `--subtitle-stream`: For video input with several subtitle streams, the stream to translate: its quoted name as listed (exact match, else unique case-insensitive prefix match) or its 1-based number (default: the only stream, or list them and exit if there are several). Ignored for `.srt` input
- `--context-windows`: Number of previous windows (original and translation) sent as context with each window; 0 disables it (default: 1)
- `--provider`: OpenRouter provider(s) to pin requests to, comma-separated in order of preference, with fallbacks to other providers disabled (sends `provider {"order": [...], "allow_fallbacks": false}`). Pass `--provider ""` to let OpenRouter route freely. Only applies to `openrouter/` models (default: `deepseek`)
- `--temperature`: Sampling temperature for the model (default: unset, so the provider/model default is used)
- `--thinking` / `--reasoning`: Model reasoning level, one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `default` (default: `none`). `none` sends OpenRouter `reasoning {"enabled": false}`, a level sends `reasoning {"effort": LEVEL}`, and `default` omits the field so OpenRouter/the model picks. Only applies to `openrouter/` models.

## How It Works

1. **Video Subtitle Extraction** (when `--input` is not a `.srt` file):
   - Uses ffmpeg to probe the video file for subtitle streams
   - Lists all available subtitle streams with language, codec, and preview text
   - Lists the streams and exits when several are present, unless `--subtitle-stream` picks one
   - Extracts the selected stream to a temporary SRT file for processing
   - Automatically cleans up temporary files after completion

2. **Windowed Processing**: The script divides the SRT file into windows of N entries (default 8). This balances between:
   - Providing enough context for the model to understand dialogue flow
   - Keeping prompts manageable and cost-effective
   - Allowing progress to be saved incrementally

3. **XML-Structured Prompts**: The run-constant instructions (target language, file description, output format) go in a single reused system prompt, and each window's subtitles are wrapped in XML tags with timing information in the user message:
   ```xml
   <file-description>Scientific documentary</file-description>
   <text id="1" start="00:00:01,000" end="00:00:03,500">Original text</text>
   <text id="2" start="00:00:04,000" end="00:00:06,200">Next subtitle</text>
   ```
   Reusing an identical system prompt across every window maximizes provider-side prompt/KV caching.

   With `--context-windows 1` (the default), the previous window is inserted before the current one as a past user/assistant exchange: its original XML as the user turn, and its translation as the assistant turn. The assistant turn is rebuilt from the parsed translations (`<answer><text id="1">...</text>...</answer>`) rather than copied from the raw reply, so it never contains the model's thinking or anything outside the `<answer>` block. A window whose translation failed is not used as context.

4. **Timing Context**: The start/end times help the model understand temporal relationships:
   - Rapid dialogue suggests conversation
   - Long pauses might indicate scene changes
   - Helps maintain natural flow in translations

5. **Retry Logic**: On a transient provider error, an unclean completion (a `finish_reason` other than `stop`/`end_turn`/`completed`, i.e. truncated or censored, or empty content), or a malformed reply (including one that does not return exactly one translated entry per input subtitle), the script:
   - Retries with exponential backoff (up to 5 attempts per window)
   - On a malformed reply, appends the error as feedback and asks the model to correct the format (leaving the cacheable prefix untouched)
   - Crashes immediately on authentication errors (retrying cannot help)
   - Falls back to original text if all attempts fail

6. **Incremental Saves**: A continuous-progress file sits next to the output with `.tmp` appended (e.g. `movie_en.srt.tmp`):
   - It is reset at the very start so a run never resumes on top of a stale `.tmp` from a previous crash
   - After each successfully translated window it is rewritten atomically (write to a sibling scratch, then `os.replace`), so an interrupt mid-write never corrupts it, it keeps the last complete window's contents
   - On completion it is atomically moved onto the final output

## Error Handling

The script provides clear error messages for common issues:

- **Network/Connection Errors**: Check your network connection and base URL
- **Authentication Errors**: Verify your API key is correct
- **Model Errors**: Ensure the model name is valid for your endpoint
- **File Not Found**: Check that the SRT or video file path is correct
- **Output Already Exists**: The script prevents accidental overwrites - remove the existing file first
- **No Subtitle Streams**: The video file doesn't contain any subtitle streams
- **FFmpeg Errors**: Ensure ffmpeg is installed and the video file is valid

## Logging

Logs are written to `./logs.txt` with:
- 10 MB rotation size
- 7 days retention
- Detailed information about processing steps and any errors

## Examples

### Translating from Video File with Multiple Subtitle Streams

```bash
uv run srt_ai_translator.py \
  --input movie.mkv \
  --output movie_en.srt \
  --target-language English
```

The tool will display available streams:
```
Found 3 subtitle streams:
  1. spa (subrip) | Preview: Este es el diálogo original...
  2. eng (subrip) - English SDH | Preview: This is the original dialogue...
  3. fre (subrip) | Preview: C'est le dialogue original...
Enter the number of the subtitle stream to translate:
```

### Translating Spanish SRT to English

```bash
uv run srt_ai_translator.py \
  --input pelicula_es.srt \
  --output movie_en.srt \
  --target-language English \
  --file-description "Mexican Spanish dialect, casual conversation"
```

### Using Different Window Sizes

For dialogue-heavy content (more context needed):
```bash
--window-size 12
```

For simple content or to reduce API costs:
```bash
--window-size 2
```

## Technical Details

- **Temperature**: Set to 0.3 for balanced consistency and natural translation
- **XML Parsing**: Uses regex-based parsing for robustness
- **Libraries**: 
  - `pysrt`: SRT file parsing and writing
  - `litellm`: Unified client for 100+ LLM providers (OpenRouter by default)
  - `tqdm`: Progress bars
  - `loguru`: Logging
  - `ffmpeg-python`: Video subtitle extraction (requires ffmpeg installed)

## Development

This project was created with [AiderBuilder](https://github.com/thiswillbeyourgithub/AiderBuilder) and developed using [aider.chat](https://github.com/Aider-AI/aider/). The migration to LiteLLM/OpenRouter and related improvements were made with [Claude Code](https://claude.com/claude-code).

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

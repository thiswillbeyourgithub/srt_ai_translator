# SRT AI Translator

Translate SRT subtitle files using OpenAI-compatible APIs with intelligent windowed processing and context awareness.

## Features

- **Video Subtitle Extraction**: Extract subtitles directly from video files with interactive stream selection
- **Windowed Translation**: Processes subtitles in configurable batches to maintain context and optimize API usage
- **OpenAI API Compatible**: Works with OpenAI API and any compatible endpoints (e.g., local LLMs via OpenAI-compatible servers)
- **Context-Aware**: Supports custom context for better translations (video type, dialect, domain-specific terminology)
- **Robust Error Handling**: Automatic retry logic with feedback to the model on parsing failures
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
pip install pysrt openai tqdm loguru ffmpeg-python
```

**Note**: For video subtitle extraction, you also need `ffmpeg` installed on your system.

## Usage

### Basic Example (SRT File)

```bash
uv run srt_ai_translator.py \
  --srt-file input.srt \
  --base-url https://api.openai.com/v1 \
  --model gpt-4 \
  --output-path output_en.srt \
  --target-language English
```

### Extract and Translate from Video

```bash
uv run srt_ai_translator.py \
  --video movie.mkv \
  --base-url https://api.openai.com/v1 \
  --model gpt-4 \
  --output-path movie_en.srt \
  --target-language English
```

If the video contains multiple subtitle streams, you'll be prompted to select which one to translate. The tool provides helpful previews to identify each stream.

### With Context and Custom Window Size

```bash
uv run srt_ai_translator.py \
  --srt-file documentary.srt \
  --base-url https://api.openai.com/v1 \
  --model gpt-4 \
  --output-path documentary_en.srt \
  --target-language English \
  --window-size 6 \
  --srt-context "Scientific documentary about marine biology, use formal terminology"
```

### Using Local LLM (e.g., via LM Studio)

```bash
export OPENAI_API_KEY="not-needed"
uv run srt_ai_translator.py \
  --srt-file movie.srt \
  --base-url http://localhost:1234/v1 \
  --model local-model \
  --output-path movie_translated.srt \
  --target-language Spanish
```

## Command-Line Arguments

### Required Arguments

- `--target-language`: Target language for translation
- `--srt-file` OR `--video`: Path to the input SRT file OR video file (exactly one must be provided)
- `--base-url`: Base URL for the OpenAI API (must start with `http://` or `https://`)
- `--model`: Model name to use for translation
- `--output-path`: Path for the output translated SRT file

### Optional Arguments

- `--api-key`: API key for OpenAI (if not provided, uses `OPENAI_API_KEY` environment variable)
- `--window-size`: Number of subtitle entries to process in each batch (default: 4)
- `--srt-context`: Context information for translation (e.g., video type, dialect, domain)

## How It Works

1. **Video Subtitle Extraction** (when using `--video`):
   - Uses ffmpeg to probe the video file for subtitle streams
   - Lists all available subtitle streams with language, codec, and preview text
   - Allows interactive selection when multiple streams are present
   - Extracts the selected stream to a temporary SRT file for processing
   - Automatically cleans up temporary files after completion

2. **Windowed Processing**: The script divides the SRT file into windows of N entries (default 4). This balances between:
   - Providing enough context for the model to understand dialogue flow
   - Keeping prompts manageable and cost-effective
   - Allowing progress to be saved incrementally

3. **XML-Structured Prompts**: Each subtitle is wrapped in XML tags with timing information:
   ```xml
   <srt-context>Scientific documentary</srt-context>
   <text id="1" start="00:00:01,000" end="00:00:03,500">Original text</text>
   <text id="2" start="00:00:04,000" end="00:00:06,200">Next subtitle</text>
   ```

4. **Timing Context**: The start/end times help the model understand temporal relationships:
   - Rapid dialogue suggests conversation
   - Long pauses might indicate scene changes
   - Helps maintain natural flow in translations

5. **Retry Logic**: If XML parsing fails, the script:
   - Includes the error message in the next prompt
   - Asks the model to correct the format
   - Retries up to 3 times per window
   - Falls back to original text if all attempts fail

6. **Incremental Saves**: After each successfully translated window:
   - Results are written to a temporary file
   - Allows recovery if the process is interrupted
   - Final output is atomically renamed on completion

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
  --video movie.mkv \
  --base-url https://api.openai.com/v1 \
  --model gpt-4 \
  --output-path movie_en.srt \
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
  --srt-file pelicula_es.srt \
  --base-url https://api.openai.com/v1 \
  --model gpt-4 \
  --output-path movie_en.srt \
  --target-language English \
  --srt-context "Mexican Spanish dialect, casual conversation"
```

### Using Different Window Sizes

For dialogue-heavy content (more context needed):
```bash
--window-size 8
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
  - `openai`: API client
  - `tqdm`: Progress bars
  - `loguru`: Logging
  - `ffmpeg-python`: Video subtitle extraction (requires ffmpeg installed)

## Development

This project was created with [AiderBuilder](https://github.com/thiswillbeyourgithub/AiderBuilder) and developed using [aider.chat](https://github.com/Aider-AI/aider/).

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

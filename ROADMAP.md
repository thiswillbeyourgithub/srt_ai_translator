# SRT AI Translator Project

## PROGRESS
- Overall completion: 0%
- TODOs remaining: 6
- Active issues: 0

## OBJECTIVES
- Create srt_ai_translator.py script that translates SRT subtitle files using OpenAI API
- Support windowed processing with configurable window size (default 4)
- Use XML format for structured prompts and responses
- Include context support for translation customization
- Implement robust error handling and retry logic for parsing failures
- Show progress with tqdm progress bar

## COMPLETED
- SETUP: Create initial ROADMAP.md structure. REASON: coordinate development across iterations. STATUS: done

## IN_PROGRESS
- IMPLEMENT: SRT file parsing using pysrt library. REASON: need to read subtitle data. STATUS: in progress

## TODO
- P0: CREATE: Basic script structure with argparse. REASON: foundation for all functionality. COMPLEXITY: low
- P0: IMPLEMENT: SRT file parsing using pysrt library. REASON: need to read subtitle data. COMPLEXITY: medium
- P0: IMPLEMENT: Windowed text processing logic. REASON: core functionality requirement. COMPLEXITY: medium
- P1: IMPLEMENT: OpenAI API integration with XML prompt format. REASON: translation engine. COMPLEXITY: high
- P1: IMPLEMENT: XML response parsing with retry logic. REASON: robust error handling. COMPLEXITY: medium
- P2: IMPLEMENT: Progress bar and output file writing. REASON: user experience. COMPLEXITY: low

## DECISIONS
- Will use pysrt library for SRT parsing (standard Python library for this purpose)
- Will use openai library for API calls
- XML format chosen for structured prompts as specified

## LESSONS_LEARNED
(none yet)

Create srt_ai_translator.py

args:
- Takes a --window-size argument that default to 4
- srt-file path to the srt file
- base-url
- model
- srt-context default to ''
- output-path

A simple script that uses whatever library you recommend to parse an srt file.

It will, using a nonsliding window of size window-size, take N "texts" of the srt file.
And prompt the llm using an openai api to translate the texts.
Use an xml like format to wrap each distinct text separately like <text id=1></text> then id=2, id=3 etc. It must ask the llm to think then use a <answer></answer> tag that contains <text id=1></text> etc. With very strict parsing. If the parsing fails me must add a message telling the model that the parsing failed and ask again to correct.

Also include the srt-context inside  the <srt-context></srt-context> xml. It can be used to specify for example the type of video, the context, some dialect like argentinian vs spanish etc.


Use tqdm for the progress bar.

then save the new srt in output-path.

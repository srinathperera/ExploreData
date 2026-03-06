import pandas as pd
import io
import re
import os
prompt_data = pd.read_json("data/filtered_prompts_with_tldr.json")
prompt_data = prompt_data[["Prompt", "tldr", "difficulty"]]

def old():
    print(prompt_data.head())

    # The delimiter we want to escape
    delimiter_literal = "<|im_start|>user\n"

    # Use re.escape()
    escaped_delimiter = re.escape(delimiter_literal) 
    # The result would be: <\|im\_start\|\>user\\n

    # You would then have to explicitly set regex=True for the split:
    prompt_data["Prompt"] = prompt_data["Prompt"].str.split(escaped_delimiter, n=1, regex=True).str[1]

    print(prompt_data.head())

def test_load_prompt_data():
    prompt_metadata_dir = "data/prompt-metadata/"
    prompt_metadata_list = []
    for filename in os.listdir(prompt_metadata_dir):
        print("filename", filename)
        if filename.endswith(".json"):
            path = os.path.join(prompt_metadata_dir, filename)
            # Robust JSON reader: supports JSONL (one JSON per line) and JSON arrays; skips blank lines
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            stripped = content.lstrip()
            try:
                if stripped.startswith("["):
                    # JSON array
                    prompt_data = pd.read_json(io.StringIO(content))
                else:
                    # JSONL; remove empty/comment lines first
                    lines = [ln for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
                    prompt_data = pd.read_json(io.StringIO("\n".join(lines)), orient='records', lines=True)
            except ValueError:
                # Fallback: try without lines flag
                prompt_data = pd.read_json(io.StringIO(content))
            if "prompt" in prompt_data.columns:
                prompt_data["Prompt"] = prompt_data["prompt"]
                #remove the Prompt column
                prompt_data = prompt_data.drop(columns=["prompt"])
            prompt_metadata_list.append(prompt_data)
    prompt_data = pd.concat(prompt_metadata_list)
    print("prompt_data.shape", prompt_data.shape, "prompt_data.columns", prompt_data.columns)
    print(prompt_data.head())
    return prompt_data


test_load_prompt_data()
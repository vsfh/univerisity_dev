import pandas as pd
import json
import os
from tqdm import tqdm
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data/feihong/hf_cache'

# --- Configuration ---
INPUT_FILE = "runs/schools.txt"
OUTPUT_FILE = "wild_university_metadata.csv"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/data/feihong/hf_cache"

# Phase 2: Strict Prompt Engineering with Köppen Standards
SYSTEM_PROMPT = """You are an expert geographer and architectural historian. 
Analyze the provided university and building. You must output ONLY valid JSON.
Do not include markdown blocks (```json), conversational text, or explanations.

Use the following schema strictly:
{
  "country": "string",
  "continent": "string",
  "estimated_area_m2": integer (Provide a heuristic estimate of the footprint area in square meters)
}
"""

#   "climate_zone": "string (Use Köppen climate classification, e.g., 'Cfa - Humid Subtropical', 'Dfb - Hemiboreal')",

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully.")


def get_metadata(school, building):
    user_prompt = f"School: {school}, Building: {building}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = (
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        or ""
    )

    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = (
        tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        or ""
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(response[start:end])
        except:
            pass
        return {
            "country": "Error",
            "continent": "Error",
            # "climate_zone": "Error",
            "estimated_area_m2": 0,
        }


# Phase 3: Python Pipeline with Checkpointing
def main():
    df_input = pd.read_csv(
        INPUT_FILE, sep=",", skipinitialspace=True, header=None, usecols=[0, 1]
    )
    df_input.columns = [
        "building",
        "school",
    ]
    if os.path.exists(OUTPUT_FILE):
        df_processed = pd.read_csv(OUTPUT_FILE)
        processed_set = set(df_processed["school"] + "_" + df_processed["building"])
        results = df_processed.to_dict("records")
        print(f"Resuming from checkpoint. {len(results)} records already processed.")
    else:
        processed_set = set()
        results = []

    for index, row in tqdm(
        df_input.iterrows(), total=len(df_input), desc="Querying Qwen"
    ):
        unique_id = f"{row['school']}_{row['building']}"

        if unique_id in processed_set:
            continue

        meta = get_metadata(row["school"], row["building"])

        combined_record = {"school": row["school"], "building": row["building"], **meta}
        results.append(combined_record)

        pd.DataFrame([combined_record]).to_csv(
            OUTPUT_FILE, mode="a", header=not os.path.exists(OUTPUT_FILE), index=False
        )


if __name__ == "__main__":
    main()

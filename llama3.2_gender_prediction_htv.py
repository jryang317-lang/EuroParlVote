import torch
import pandas as pd
import argparse
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Updated prompt template to allow "Unknown"
PROMPT_TEMPLATE = '''You are a language expert who can infer the likely gender of a speaker based on their written text.
Analyze the style, word choice, and other linguistic cues in the text to decide if the speaker is male, female, or unknown.
If you are uncertain based on the text, you should answer "Unknown".

<MEP_speech>
{speech}
</MEP_speech>

Provide:
1. Your prediction ("Male", "Female", or "Unknown")
2. A confidence score on a scale of 1-5
3. Reasoning for your prediction

At the end, respond only with the final answer in this JSON format in English, prefixed with "The final answer is:".

{{
"gender": "Male/Female/Unknown",
"confidence_level": N,
"reasoning": "Your detailed reasoning here"
}}

'''

def extract_gender_from_raw_output(raw_output):
    try:
        final_answer_match = re.search(r"The final answer is:\s*({[\s\S]+?})", raw_output)
        if final_answer_match:
            json_block = final_answer_match.group(1)
            gender_match = re.search(r'"gender"\s*:\s*"([^"]+)"', json_block)
            if gender_match:
                gender = gender_match.group(1).strip()
                if gender not in ["Male", "Female", "Unknown"]:
                    return "", f"Invalid gender value '{gender}'"
                return gender, None
            else:
                return "", "Gender field not found in JSON block"
        else:
            return "", "No 'The final answer is:' block found"
    except Exception as e:
        return "", f"Regex extract failed: {e}"

def predict_gender(model, tokenizer, speech, temperature=0.0):
    prompt = PROMPT_TEMPLATE.format(speech=speech)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False if temperature == 0.0 else True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gender, error = extract_gender_from_raw_output(generated_text)
    return {
        "raw_output": generated_text,
        "model_gender": gender,
        "error": error
    }

def main(args):
    df = pd.read_csv(args.input_csv)

    if args.num_samples > 0:
        df = df.sample(n=args.num_samples, random_state=42)

    # Load model/tokenizer
    custom_cache_dir = "/nas02/jiryang/llama32"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=custom_cache_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    predicted_genders = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating gender predictions"):
        speech = str(row['Speech'])
        prediction_info = predict_gender(model, tokenizer, speech, args.temperature)

        predicted_gender = prediction_info["model_gender"]
        predicted_genders.append(predicted_gender)

        output_row = {
            "Report_ID": row.get("Report_ID", None),
            "MEP_ID": row.get("MEP_ID", None),
            "Speech": speech,
            "ideal_gender": row.get('gender', "Unknown"),
            "model_output": {
                "raw_output": prediction_info["raw_output"],
                "model_gender": prediction_info["model_gender"],
                "error": prediction_info["error"]
            }
        }

        results.append(output_row)
        if args.debug:
            print(json.dumps(output_row, indent=2))
            print("=" * 80)

    # Save output JSON
    model_tag = "llama32"
    out_json_path = args.output_json or f"gender_predictions_{model_tag}_{args.num_samples}.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Predictions saved to {out_json_path}")

    # Save output CSV (add predicted_gender column)
    df['predicted_gender'] = predicted_genders
    out_csv_path = args.output_csv or f"gender_predictions_{model_tag}_{args.num_samples}.csv"
    df.to_csv(out_csv_path, index=False)
    print(f"✅ CSV saved with 'predicted_gender' column to {out_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the CSV file with MEP speeches.")
    parser.add_argument("--output_json", type=str, required=False, help="Path to save the output JSON file.")
    parser.add_argument("--output_csv", type=str, required=False, help="Path to save the output CSV file with predicted gender.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name or path of the model.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature.")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of random samples to use from the CSV. 0 means use all samples.")
    parser.add_argument("--debug", action="store_true", help="Display debug output for each prediction.")
    args = parser.parse_args()
    main(args)

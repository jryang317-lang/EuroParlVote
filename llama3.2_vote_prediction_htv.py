import torch
import pandas as pd
import argparse
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM



PROMPT_TEMPLATE = '''You will analyze EU parliament MEP's speeches and  debate topics to predict voting outcomes with probability assessments.

Review the following debate topic and MEP demographic information:

<debate_topic>
{title}
</debate_topic>

<MEP_speech>
{speech}
</MEP_speech>

{hint}

Analyze the content and provide:
1. Your prediction ('For' for positive support, or 'Against' for negative reject)
2. Probability scores for both outcomes (must sum to 100%)
3. Confidence level on a scale of 1-5
4. Reasoning for your prediction

At the end, response only the final answer in this JSON format below in English, with prefix 'The final answer is:'.
{{
    "prediction": "For/Against",
    "probabilities": {{
        "For": "X%",
        "Against": "Y%"
    }},
    "confidence_level": N,
    "reasoning": "Your detailed reasoning here"
}}

'''


def extract_prediction_from_raw_output(raw_output):
    try:
        # Locate the block starting with "The final answer is:"
        final_answer_match = re.search(r"The final answer is:\s*({[\s\S]+?})", raw_output)
        if final_answer_match:
            json_block = final_answer_match.group(1)
            # Find the prediction string using regex
            pred_match = re.search(r'"prediction"\s*:\s*"([^"]+)"', json_block)
            if pred_match:
                return pred_match.group(1), None
            else:
                return "", "Prediction field not found in JSON block"
        else:
            return "", "No 'The final answer is:' block found"
    except Exception as e:
        return "", f"Regex extract failed: {e}"


def predict_vote(model, tokenizer, title, speech, hint, temperature=0.0):
    prompt = PROMPT_TEMPLATE.format(title=title, speech=speech, hint=hint)
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
    prediction, error = extract_prediction_from_raw_output(generated_text)
    return {
        "raw_output": generated_text,
        "model_prediction": prediction,
        "error": error
    }

def main(args):
    df = pd.read_csv(args.input_csv)

    # Compute age
    df['age'] = 2025 - pd.to_datetime(df['date_of_birth'], dayfirst=True, errors='coerce').dt.year

    if args.num_samples > 0:
        df = df.sample(n=args.num_samples, random_state=42)

    # Load model/tokenizer
    custom_cache_dir = "/nas02/jiryang/llama32"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=custom_cache_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    GROUP_MAPPING = {
        "RENEW": "Renew Europe Group",
        "SD": "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament",
        "EPP": "Group of the European People’s Party",
        "GREEN_EFA": "Group of the Greens/European Free Alliance",
        "GUE_NGL": "The Left group in the European Parliament – GUE/NGL",
        "ECR": "European Conservatives and Reformists Group",
        "NI": "Non-attached Members",
        "PFE": "Patriots for Europe Group",
        "ID": "Identity and Democracy Group",
        "ESN": "Europe of Sovereign Nations Group"
    }


    # Context hint logic
    context_map = {
        "basic": "",
        "with_gender": lambda row: f"The speaker is a {row.get('gender', 'unknown').lower()} MEP.",
        "with_age": lambda row: f"The speaker is approximately {row.get('age', 'unknown')} years old.",
        "with_country": lambda row: f"The speaker is from {row.get('country_code_x', 'unknown')}.",
        "with_group": lambda
            row: f"The speaker is from {row.get('group_code', 'unknown')}: {GROUP_MAPPING.get(row.get('group_code', 'unknown'), 'Unknown Group')} political group."
    }

    if args.context_type not in context_map:
        raise ValueError(f"Invalid context_type '{args.context_type}'. Must be one of {list(context_map.keys())}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
        title = str(row['Title_EN'])
        speech = str(row['Speech'])
        vote_label = row.get('position', 'Unknown')
        gender = row.get('gender', 'Unknown')
        age = row.get('age', 'Unknown')
        country = row.get('country_code_x', 'Unknown')
        group = row.get('group_code', 'Unknown')
        hint = context_map[args.context_type](row) if callable(context_map[args.context_type]) else context_map[args.context_type]

        prediction_info = predict_vote(model, tokenizer, title, speech, hint, args.temperature)

        output_row = {
            "Report_ID": row["Report_ID"],
            "MEP_ID": int(row["MEP_ID"]),
            "Title_EN": title,
            "Speech": speech,
            "ideal_output": vote_label,
            "gender": gender,
            "age": age,
            "country": country,
            "group": group,
            "context_type": args.context_type,
            "model_output": {
                "raw_output": prediction_info["raw_output"],
                "model_prediction": prediction_info["model_prediction"],
                "error": prediction_info["error"]
            }
        }

        results.append(output_row)
        if args.debug:
            print(json.dumps(output_row, indent=2))
            print("=" * 80)

    # Save output
    model_tag = "llama32"
    out_path = args.output_json or f"vote_predictions_{model_tag}_{args.context_type}_{args.num_samples}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Predictions saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=False)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--context_type", type=str, default="basic", choices=["basic", "with_gender", "with_age", "with_country", "with_group"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

import argparse
import pandas as pd
import openai
import json
import time
from tqdm import tqdm
from datetime import datetime
import csv
import re

openai.api_key = ''  # ← your API key

PROMPT_TEMPLATE = '''You are a language expert who can infer the likely gender of a speaker based on their written text.
Analyze the style, word choice, and other linguistic cues in the text to decide if the speaker is male, female.

<MEP_speech>
{speech}
</MEP_speech>

{hint}

Provide:
1. Your prediction ("Male", "Female")
2. A confidence score on a scale of 1-5
3. Reasoning for your prediction

At the end, respond only with the final answer in this JSON format in English.

{{
"gender": "Male/Female",
"confidence_level": N,
"reasoning": "Your detailed reasoning here"
}}
'''


def extract_fields_from_output(output):
    try:
        cleaned = output.strip("```json").strip("```")
        parsed = json.loads(cleaned)
        return (
            parsed.get("gender", ""),
            parsed.get("confidence_level", ""),
            parsed.get("reasoning", ""),
            None
        )
    except Exception as e:
        return "", "", "", f"Parsing error: {e}"


def predict_gender(model_name, speech, hint, temperature=0.3):
    prompt = PROMPT_TEMPLATE.format(speech=speech, hint=hint)
    messages = [
        {"role": "system", "content": "You are a political analysis assistant."},
        {"role": "user", "content": prompt}
    ]
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=512
            )
            return response.choices[0].message["content"]
        except openai.error.RateLimitError:
            print("Rate limited, retrying...")
            time.sleep(30)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    return "Error"


def main(args):
    df = pd.read_csv(args.input_csv)
    df['age'] = 2025 - pd.to_datetime(df['date_of_birth'], dayfirst=True, errors='coerce').dt.year
    if args.num_samples > 0:
        df = df.sample(n=args.num_samples, random_state=args.seed)

    GROUP_MAPPING = {
        "RENEW": "Renew Europe Group",
        "SD": "Progressive Alliance of Socialists and Democrats",
        "EPP": "European People’s Party",
        "GREEN_EFA": "Greens/European Free Alliance",
        "GUE_NGL": "The Left – GUE/NGL",
        "ECR": "European Conservatives and Reformists",
        "NI": "Non-attached Members",
        "PFE": "Patriots for Europe Group",
        "ID": "Identity and Democracy Group",
        "ESN": "Europe of Sovereign Nations Group"
    }

    context_map = {
        "basic": "",
        "with_gender": lambda row: f"The speaker is a {row.get('gender', 'unknown').lower()} MEP.",
        "with_age": lambda row: f"The speaker is approximately {row.get('age', 'unknown')} years old.",
        "with_country": lambda row: f"The speaker is from {row.get('country_code_x', 'unknown')}.",
        "with_group": lambda row: f"The speaker is from {row.get('group_code', 'unknown')}: {GROUP_MAPPING.get(row.get('group_code', 'unknown'), 'Unknown Group')}."
    }

    if args.context_type not in context_map:
        raise ValueError(f"Invalid context_type '{args.context_type}'.")

    json_results = []
    csv_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = row.get('Title_EN', '')
        speech = row.get('Speech', '')
        gender = row.get('gender', 'Unknown')
        age = row.get('age', 'Unknown')
        country = row.get('country_code_x', 'Unknown')
        group = row.get('group_code', 'Unknown')
        hint = context_map[args.context_type](row) if callable(context_map[args.context_type]) else context_map[args.context_type]

        raw_output = predict_gender(args.model_name, speech, hint, args.temperature)
        predicted_gender, confidence_level, reasoning, error = extract_fields_from_output(raw_output)

        result_json = {
            "Report_ID": row.get('Report_ID'),
            "MEP_ID": row.get('MEP_ID'),
            "Title_EN": title,
            "Speech": speech,
            "ideal_output": gender,
            "gender": gender,
            "age": age,
            "country": country,
            "group": group,
            "context_type": args.context_type,
            "model_output": raw_output,
            "predicted_gender": predicted_gender,
            "confidence_level": confidence_level,
            "reasoning": reasoning,
            "error": error
        }
        json_results.append(result_json)

        csv_rows.append({
            "Report_ID": row.get('Report_ID'),
            "MEP_ID": row.get('MEP_ID'),
            "true_gender": gender,
            "predicted_gender": predicted_gender,
            "confidence_level": confidence_level,
            "reasoning": reasoning,
            "age": age,
            "country": country,
            "group": group,
            "context_type": args.context_type
        })

        if args.debug:
            print(json.dumps(result_json, indent=2))
            print("=" * 80)

    json_out_path = args.output_json or f"gender_predictions_{args.model_name}_{args.context_type}_{args.num_samples}.json"
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON predictions saved to {json_out_path}")

    if args.output_csv:
        csv_header = [
            "Report_ID", "MEP_ID", "true_gender", "predicted_gender",
            "confidence_level", "reasoning", "age", "country", "group", "context_type"
        ]
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"📄 CSV predictions saved to {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict MEP gender using GPT model with demographic context")
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_json', type=str, default=None, help='Path to output JSON file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to output CSV file')
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--context_type', type=str, default='basic',
                        choices=['basic', 'with_gender', 'with_age', 'with_country', 'with_group'])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

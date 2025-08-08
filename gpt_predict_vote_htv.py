import argparse
import pandas as pd
import openai
import json
import time
import re
import csv
from tqdm import tqdm
from datetime import datetime

# Set your OpenAI API key
openai.api_key = ''  # replace with your real key

PROMPT_TEMPLATE = '''You will analyze EU parliament MEP's speeches and debate topics to predict voting outcomes with probability assessments.

Review the following debate topic and MEP demographic information:

<debate_topic>
{title}
</debate_topic>

<MEP_speech>
{speech}
</MEP_speech>


{demographic_hint}

Analyze the content and provide:
1. Your prediction ('For' for positive support, or 'Against' for negative reject)
2. Probability scores for both outcomes (must sum to 100%)
3. Confidence level on a scale of 1-5, where:
   1 = Very low confidence (highly uncertain)
   2 = Low confidence (somewhat uncertain)
   3 = Moderate confidence (balanced certainty)
   4 = High confidence (reasonably certain)
   5 = Very high confidence (highly certain)
4. Reasoning for your prediction

Format your response in JSON with the following structure:
{{
    "prediction": "For/Against",
    "probabilities": {{
        "For": "X%",
        "Against": "Y%"
    }},
    "confidence_level": N,
    "reasoning": "Your detailed reasoning here"
}}'''


def extract_final_answer(raw_output):
    try:
        match = re.search(r'```json\n({.*?})\n```', raw_output, re.DOTALL)
        if match:
            final_json = json.loads(match.group(1))
            return (
                final_json.get("prediction", ""),
                final_json.get("probabilities", {}).get("For", ""),
                final_json.get("probabilities", {}).get("Against", ""),
                final_json.get("confidence_level", ""),
                final_json.get("reasoning", "")
            )
    except (json.JSONDecodeError, IndexError):
        return "", "", "", "", ""
    return "", "", "", "", ""


def predict_vote(model_name, title, speech, demographic_hint, temperature=0.3):
    prompt = PROMPT_TEMPLATE.format(title=title, speech=speech, demographic_hint=demographic_hint)
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
            print("Rate limited, waiting 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(5)
    return "Error"


def main(args):
    df = pd.read_csv(args.input_csv)
    df['age'] = 2025 - pd.to_datetime(df['date_of_birth'], dayfirst=True, errors='coerce').dt.year
    if args.num_samples > 0:
        df = df.sample(n=args.num_samples, random_state=args.seed)

    results = []
    csv_rows = []

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

    context_map = {
        "basic": "",
        "with_gender": lambda row: f"The speaker is a {row.get('gender', 'unknown').lower()} MEP.",
        "with_age": lambda row: f"The speaker is approximately {row.get('age', 'unknown')} years old.",
        "with_country": lambda row: f"The speaker is from {row.get('country_code_x', 'unknown')}.",
        "with_group": lambda row: f"The speaker is from {row.get('group_code', 'unknown')}: {GROUP_MAPPING.get(row.get('group_code', 'unknown'), 'Unknown Group')} political group."
    }

    if args.context_type not in context_map:
        raise ValueError(f"Invalid context_type '{args.context_type}'. Must be one of {list(context_map.keys())}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = row['Title_EN']
        speech = row['Speech']
        vote_label = row.get('position', 'Unknown')
        gender = row.get('gender', 'Unknown')
        age = row.get('age', 'Unknown')
        country = row.get('country_code_x', 'Unknown')
        group = row.get('group_code', 'Unknown')
        hint = context_map[args.context_type](row) if callable(context_map[args.context_type]) else context_map[args.context_type]

        model_output = predict_vote(args.model_name, title, speech, hint, args.temperature)

        prediction, prob_for, prob_against, confidence, reasoning = extract_final_answer(model_output)

        results.append({
            "Report_ID": row['Report_ID'],
            "MEP_ID": row['MEP_ID'],
            "Title_EN": title,
            "Speech": speech,
            "ideal_output": vote_label,
            "gender": gender,
            "age": age,
            "country": country,
            "group": group,
            "context_type": args.context_type,
            "model_output": model_output
        })

        csv_rows.append({
            "Report_ID": row['Report_ID'],
            "MEP_ID": row['MEP_ID'],
            "Title_EN": title,
            "Speech": speech,
            "ideal_output": vote_label,
            "gender": gender,
            "age": age,
            "country": country,
            "group": group,
            "context_type": args.context_type,
            "prediction": prediction,
            "probability_for": prob_for,
            "probability_against": prob_against,
            "confidence_level": confidence,
            "reasoning": reasoning
        })

        if args.debug:
            print(json.dumps(results[-1], indent=2))
            print("=" * 50)

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Predictions saved to {args.output_json}")

    csv_header = [
        "Report_ID", "MEP_ID", "Title_EN", "Speech", "ideal_output", "gender",
        "age", "country", "group", "context_type",
        "prediction", "probability_for", "probability_against",
        "confidence_level", "reasoning"
    ]

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"📄 CSV version saved to {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict MEP vote using GPT model with demographic context")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_json', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='OpenAI model name')
    parser.add_argument('--temperature', type=float, default=0.3, help='Sampling temperature')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of rows to process')
    parser.add_argument('--debug', action='store_true', help='Print responses for inspection')
    parser.add_argument('--context_type', type=str, required=True,
                        choices=['basic', 'with_gender', 'with_age', 'with_country', 'with_group'],
                        help='Which context type to use in prompt')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

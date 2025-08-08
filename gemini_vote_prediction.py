import pandas as pd
import argparse
import json
import re
from tqdm import tqdm
from google import genai
import csv

PROMPT_TEMPLATE = '''You will analyze EU parliament MEP's speeches and debate topics to predict voting outcomes with probability assessments.

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

At the end, response only the final answer in this JSON format below in English.
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
        cleaned_output = raw_output.strip("```json").strip("```")
        model_output = json.loads(cleaned_output)
        prediction = model_output.get("prediction", "").strip()
        if prediction:
            return prediction, None
        else:
            return "", "Prediction field not found in model output"
    except Exception as e:
        print(f"Error parsing model_output: {e}")
        return "", f"JSON parsing failed: {e}"


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


def predict_vote_gemini(client, title, speech, hint, model_name):
    prompt = PROMPT_TEMPLATE.format(title=title, speech=speech, hint=hint)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    generated_text = response.text
    prediction, error = extract_prediction_from_raw_output(generated_text)
    return {
        "raw_output": generated_text,
        "model_prediction": prediction,
        "error": error
    }


def main(args):
    client = genai.Client(api_key="")
    df = pd.read_csv(args.input_csv)
    if args.num_samples > 0:
        df = df.sample(n=args.num_samples, random_state=42)
    results = []
    csv_rows = []

    context_map = {
        "basic": "",
        "with_gender": lambda row: f"The speaker is a {row.get('gender', 'unknown').lower()} MEP.",
        "with_age": lambda row: f"The speaker is approximately {row.get('age', 'unknown')} years old.",
        "with_country": lambda row: f"The speaker is from {row.get('country_code_x', 'unknown')}.",
        "with_group": lambda row: f"The speaker is from {row.get('group_code', 'unknown')} political group."
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
        title = str(row['Title_EN'])
        speech = str(row['Speech'])
        hint_fn = context_map.get(args.context_type)
        hint = hint_fn(row) if callable(hint_fn) else hint_fn

        prediction_info = predict_vote_gemini(client, title, speech, hint, args.model_name)

        age_val = (
            2025 - pd.to_datetime(row.get('date_of_birth', '1900-01-01'), errors='coerce').year
            if pd.notnull(row.get('date_of_birth')) else 'Unknown'
        )

        output_row = {
            "Report_ID": row["Report_ID"],
            "MEP_ID": int(row["MEP_ID"]),
            "Title_EN": title,
            "Speech": speech,
            "model_output": prediction_info,
            "ideal_output": row.get('position', 'Unknown'),
            "gender": row.get('gender', 'Unknown'),
            "age": age_val,
            "country": row.get('country_code_x', 'Unknown'),
            "group": row.get('group_code', 'Unknown')
        }
        results.append(output_row)

        raw_output = prediction_info.get("raw_output", "")
        prediction, probability_for, probability_against, confidence_level, reasoning = extract_final_answer(raw_output)

        csv_rows.append({
            "Report_ID": output_row["Report_ID"],
            "MEP_ID": output_row["MEP_ID"],
            "Title_EN": output_row["Title_EN"],
            "Speech": output_row["Speech"],
            "ideal_output": output_row["ideal_output"],
            "gender": output_row["gender"],
            "age": output_row["age"],
            "country": output_row["country"],
            "group": output_row["group"],
            "model_prediction": prediction_info["model_prediction"],
            "raw_output": raw_output,
            "confidence_level": confidence_level,
            "probability_for": probability_for,
            "probability_against": probability_against,
            "reasoning": reasoning
        })

        if args.debug:
            print(json.dumps(output_row, indent=2))

    with open(args.output_json, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, indent=2, ensure_ascii=False)
    print(f"✅ Predictions saved to {args.output_json}")

    csv_header = [
        "Report_ID", "MEP_ID", "Title_EN", "Speech", "ideal_output", "gender",
        "age", "country", "group",
        "model_prediction", "raw_output", "confidence_level",
        "probability_for", "probability_against", "reasoning"
    ]
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"📄 CSV version saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="models/gemini-2.5-flash-preview-04-17")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--context_type", type=str, default="basic", choices=["basic", "with_gender", "with_age", "with_country", "with_group"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

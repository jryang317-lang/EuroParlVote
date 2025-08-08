import pandas as pd
import argparse
import json
import re
from tqdm import tqdm
from google import genai
import csv

PROMPT_TEMPLATE = '''You will analyze a speech made by a Member of the European Parliament (MEP) and predict the gender of the speaker.

<MEP_speech>
{speech}
</MEP_speech>

{hint}

Predict the speaker's gender: 'Male' or 'Female'.

At the end, respond **only** in the following JSON format:
{{
  "prediction": "Male/Female",
  "confidence_level": N,  // from 1 (low) to 5 (high)
  "reasoning": "Your detailed reasoning here"
}}
'''


def extract_gender_prediction(raw_output):
    try:
        cleaned_output = raw_output.strip("```json").strip("```")
        model_output = json.loads(cleaned_output)
        prediction = model_output.get("prediction", "").strip()
        return (
            prediction,
            model_output.get("confidence_level", ""),
            model_output.get("reasoning", ""),
            None
        )
    except Exception as e:
        return "", "", "", f"Parsing error: {e}"


def predict_gender_gemini(client, speech, hint, model_name):
    prompt = PROMPT_TEMPLATE.format(speech=speech, hint=hint)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    generated_text = response.text
    return extract_gender_prediction(generated_text), generated_text


def main(args):
    client = genai.Client(api_key="")
    df = pd.read_csv(args.input_csv)
    if args.num_samples > 0:
        df = df.sample(n=args.num_samples, random_state=42)

    context_map = {
        "basic": "",
        "with_age": lambda row: f"The speaker is approximately {row.get('age', 'unknown')} years old.",
        "with_country": lambda row: f"The speaker is from {row.get('country_code_x', 'unknown')}.",
        "with_group": lambda row: f"The speaker is affiliated with the {row.get('group_code', 'unknown')} political group."
    }

    results = []
    csv_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting gender"):
        speech = str(row.get("Speech", ""))
        hint_fn = context_map.get(args.context_type)
        hint = hint_fn(row) if callable(hint_fn) else hint_fn

        (pred, conf_lvl, reason, error), raw_output = predict_gender_gemini(
            client, speech, hint, args.model_name
        )

        age_val = (
            2025 - pd.to_datetime(row.get('date_of_birth', '1900-01-01'), errors='coerce').year
            if pd.notnull(row.get('date_of_birth')) else 'Unknown'
        )

        output_row = {
            "Report_ID": row.get("Report_ID", ""),
            "MEP_ID": int(row.get("MEP_ID", -1)),
            "Speech": speech,
            "predicted_gender": pred,
            "confidence_level": conf_lvl,
            "reasoning": reason,
            "raw_output": raw_output,
            "true_gender": row.get("gender", "Unknown"),
            "age": age_val,
            "country": row.get("country_code_x", "Unknown"),
            "group": row.get("group_code", "Unknown"),
            "error": error
        }

        results.append(output_row)

        csv_rows.append({
            "Report_ID": output_row["Report_ID"],
            "MEP_ID": output_row["MEP_ID"],
            "true_gender": output_row["true_gender"],
            "predicted_gender": output_row["predicted_gender"],
            "confidence_level": output_row["confidence_level"],
            "reasoning": output_row["reasoning"],
            "age": output_row["age"],
            "country": output_row["country"],
            "group": output_row["group"],
            "raw_output": raw_output
        })

        if args.debug:
            print(json.dumps(output_row, indent=2))

    with open(args.output_json, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, indent=2, ensure_ascii=False)
    print(f"✅ JSON output saved to {args.output_json}")

    csv_header = [
        "Report_ID", "MEP_ID", "true_gender", "predicted_gender", "confidence_level",
        "reasoning", "age", "country", "group", "raw_output"
    ]
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"📄 CSV output saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="models/gemini-2.5-flash-preview-04-17")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--context_type", type=str, default="basic", choices=["basic", "with_age", "with_country", "with_group"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

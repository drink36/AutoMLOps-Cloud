import os
import argparse
import io

from inference import model_fn, input_fn, predict_fn, output_fn


def main():
    parser = argparse.ArgumentParser(description="Local test for SageMaker inference logic")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the directory containing model_artifact.pkl (and model.tar.gz if used)")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to a sample input file (CSV or JSON) to test inference")
    parser.add_argument("--accept", type=str, choices=["csv", "json"], default="csv",
                        help="Response format: 'csv' or 'json'")
    args = parser.parse_args()

    # Load the trainer with the trained models
    print(f"Loading model from: {args.model_dir}")
    trainer = model_fn(args.model_dir)

    # Read raw input
    with open(args.input_file, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Determine content type by extension
    ext = os.path.splitext(args.input_file)[1].lower()
    if ext == ".csv":
        content_type = "text/csv"
    elif ext == ".json":
        content_type = "application/json"
    else:
        raise ValueError(f"Unsupported input file extension: {ext}")

    # Parse input into DataFrame
    print(f"Parsing input file as: {content_type}")
    data = input_fn(raw, content_type)
    print("Input DataFrame:")
    print(data.head())

    # Run prediction
    print("Running prediction...")
    preds = predict_fn(data, trainer)
    print("Raw prediction output:")
    print(preds.head())

    # Format output
    accept_header = f"text/{args.accept}" if args.accept == "csv" else "application/json"
    print(f"Formatting output as: {args.accept}")
    output = output_fn(preds, accept_header)
    print("Formatted output:")
    print(output)


if __name__ == "__main__":
    main()
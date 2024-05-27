import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default="", help="path to the originial output json.")
    parser.add_argument("--output_file_path", type=str, default="", help="path to where you want to save the processed json.")
    args = parser.parse_args()
    
    with open(args.input_file_path , 'r') as f:
        data = json.load(f)

    transformed_data = [{"questionId": item["question_id"], "answer": item["answer"].replace("</s>", "")} for item in data]

    with open(args.output_file_path, 'w') as f:
        json.dump(transformed_data, f)

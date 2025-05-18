import json
import argparse


def process_jsonl(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                data = json.loads(line.strip())
                # check if 'generations' is in the data and is a list
                if 'generations' in data and isinstance(data['generations'], list):
                    for gen in data['generations']:
                        if isinstance(gen, dict) and 'text' in gen:
                            # extract the 'text' field
                            output_obj = {"text": gen["text"]}
                            outfile.write(json.dumps(
                                output_obj, ensure_ascii=False) + '\n')

        print(f"Finished, data saved to {output_path}")

    except FileNotFoundError:
        print(f"Fail, no such file: {input_path}")
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
    except Exception as e:
        print(f"Unknown error: {e}")


def main():
    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        help='input jsonl filepath')
    parser.add_argument(
        '--output', '-o', default='output.jsonl', help='output jsonl filepath (default: output.jsonl)')
    args = parser.parse_args()
    process_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()

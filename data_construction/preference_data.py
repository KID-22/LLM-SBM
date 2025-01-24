import pandas as pd
from typing import Optional
import json
from tqdm import tqdm
import sys, os
import argparse
import logging
import copy
import re
import csv
from sentence_transformers import SentenceTransformer

def load_result(save_path, dataset, model, llm_name):
    with open(f'{save_path}/{dataset}/{model}/human-{llm_name}.json', 'r') as f:
        data = json.load(f)
    return data


def load_qrels(qrels_path):
    qrels = {}
    reader = csv.reader(open(f"{qrels_path}", encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score

    return qrels


def load_corpus(data_path):
    corpus = {}
    
    num_lines = sum(1 for i in open(data_path, 'rb'))
    with open(data_path, encoding='utf8') as fIn:
        for line in tqdm(fIn, total=num_lines):
            line = json.loads(line)
            corpus[line.get("_id")] = {
                "text": line.get("text", ""),
                "title": line.get("title", ""),
                "metadata": line.get("metadata", {}),
            }
    return corpus


def get_relevant_result(result, qrels, llm_name):
    rank_result = {}
    rank_result["human"] = {}
    rank_result["llm"] = {}
    score_result = {}
    score_result["human"] = {}
    score_result["llm"] = {}
    for query_id, res in result.items():
        for key in qrels[query_id].keys():
            sorted_keys = sorted(res, key=res.get, reverse=True)
            rank_dict = {}
            for rank, k in enumerate(sorted_keys):
                rank_dict[k] = rank + 1
            if key + "-human" in res.keys():
                rank_result["human"][query_id + "@@@" + key] = rank_dict[key + "-human"]
                score_result["human"][query_id + "@@@" + key] = res[key + "-human"]
            else:
                rank_result["human"][query_id + "@@@" + key] = None
                score_result["human"][query_id + "@@@" + key] = None
            if key + f"-{llm_name}" in res.keys():
                rank_result["llm"][query_id + "@@@" + key] = rank_dict[key + f"-{llm_name}"]
                score_result["llm"][query_id + "@@@" + key] = res[key + f"-{llm_name}"]
            else:
                rank_result["llm"][query_id + "@@@" + key] = None
                score_result["llm"][query_id + "@@@" + key] = None

    return rank_result, score_result


def avg_rank_and_score_diff(retriever_result_list, qrels, llm_name):
    rank_result_list = []
    score_result_list = []
    for result in retriever_result_list:
        rank_result, score_result = get_relevant_result(result, qrels, llm_name)
        rank_result_list.append(rank_result)
        score_result_list.append(score_result)
    
    
    avg_rank_diff_dict = {}
    avg_score_diff_dict = {}
    for query_id in qrels.keys():
        for doc_id in qrels[query_id].keys():
            rank_diff = 0
            score_diff = 0
            for rank_result, score_result in zip(rank_result_list, score_result_list):
                if rank_result["llm"][query_id + "@@@" + doc_id] is not None and rank_result["human"][query_id + "@@@" + doc_id] is not None:
                    rank_diff += rank_result["llm"][query_id + "@@@" + doc_id] - rank_result["human"][query_id + "@@@" + doc_id]
                if score_result["llm"][query_id + "@@@" + doc_id] is not None and score_result["human"][query_id + "@@@" + doc_id] is not None:
                    score_diff += score_result["llm"][query_id + "@@@" + doc_id] - score_result["human"][query_id + "@@@" + doc_id]
            avg_rank_diff_dict[query_id + "@@@" + doc_id] = rank_diff / len(rank_result_list)
            avg_score_diff_dict[query_id + "@@@" + doc_id] = score_diff / len(score_result_list)
    
    return avg_rank_diff_dict, avg_score_diff_dict


def apply_template(prompt_data, preference_samples, human_text, chosen_text, rejected_text, prefer_diff):
    instruction = prompt_data["instruction"]
    instruction += f"""Here is the original text: {human_text}\n\nPlease give your answer in the following format: {{"Rewritten Text": "Your generated text"}}.\n"""
    system = prompt_data["system"]
    sample = {
        "conversations": [
            {
                "from": "system",
                "value": system,
            },
            {
                "from": "human",
                "value": instruction,
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": f"""{{"Rewritten Text": "{chosen_text}"}}""",
        },
        "rejected": {
            "from": "gpt",
            "value": f"""{{"Rewritten Text": "{rejected_text}"}}""",
        },
        "weight": prefer_diff
    }
    preference_samples.append(sample)
    
    return preference_samples


def normalize_prefer_diff(preference_samples, min_diff, max_diff):
    # Avoid division by zero
    min_diff = 0
    if max_diff == min_diff:
        for sample in preference_samples:
            sample['weight'] = 1  # Or any constant value
    else:
        for sample in preference_samples:
            sample['weight'] = (sample['weight'] - min_diff) / (max_diff - min_diff)
    return preference_samples


def compute_similarity(bge_model, sentences_1, sentences_2):
    embeddings_1 = bge_model.encode(sentences_1, show_progress_bar=False, normalize_embeddings=True)
    embeddings_2 = bge_model.encode(sentences_2, show_progress_bar=False, normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    return similarity


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--result_path', type=str, help='path to the result')
    parser.add_argument('--data_path', type=str, help='path to the data')
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument('--threshold', type=float, help='threshold', default=0.0)
    parser.add_argument('--output_path', type=str, help='path to the output')
    parser.add_argument('--llm_name', type=str, help='llm_name')
    parser.add_argument('--prompt_id_list', nargs='+', type=str, help='prompt file name list')
    parser.add_argument('--select_by', type=str, choices=['rank', 'score'], default='rank', help='Method to select data')  # New argument added here
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    #### print args
    logging.info("result_path: %s", args.result_path)
    logging.info("data_path: %s", args.data_path)
    logging.info("dataset: %s", args.dataset)
    logging.info("threshold: %s", args.threshold)
    logging.info("output_path: %s", args.output_path)
    logging.info("llm_name: %s", args.llm_name)
    logging.info("prompt_id_list: %s", args.prompt_id_list)
    logging.info("select_by: %s", args.select_by)  # Log the new argument

    # Load data
    corpus = {}
    prompt_data_list = []
    llm_name_list = []
    human_corpus = load_corpus(f"{args.data_path}/{args.dataset}/corpus/human.jsonl")
    corpus["human"] = human_corpus
    for prompt_id in args.prompt_id_list:
        llm_name_list.append(prompt_id + "-" + args.llm_name)
        corpus[prompt_id + "-" + args.llm_name] = load_corpus(f"{args.data_path}/{args.dataset}/corpus/{prompt_id}-{args.llm_name}.jsonl")
        prompt_data_list.append(json.load(open(f"prompts/{prompt_id}.json")))
    
    args.output_path = f"{args.output_path}/{args.dataset}"
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    retriever_result = {}
    for llm_name in llm_name_list:
        retriever_result_list = []
        for ir_model in ["bert-base-uncased-mean-v3-msmarco"]:
            retriever_result_list.append(load_result(args.result_path, args.dataset, ir_model, llm_name))
        retriever_result[llm_name] = retriever_result_list

    if args.dataset == "msmarco":
        qrels_path = f"{args.data_path}/{args.dataset}/qrels/dev.tsv"
    else:
        qrels_path = f"{args.data_path}/{args.dataset}/qrels/test.tsv"
    qrels = load_qrels(qrels_path)

    avg_rank_diff_dict = {}
    avg_score_diff_dict = {}
    for llm_name in llm_name_list:
        avg_rank_diff_dict[llm_name], avg_score_diff_dict[llm_name] = avg_rank_and_score_diff(retriever_result[llm_name], qrels, llm_name)
    
    bge_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    preference_samples = []
    count = 0
    max_diff = float('-inf')
    min_diff = float('inf')
    for key in avg_rank_diff_dict[llm_name_list[0]].keys():
        for llm1 in llm_name_list:
            for llm2 in llm_name_list:
                if llm1 == llm2:
                    continue
                # Skip if contains Chinese characters
                if re.search("[\u4e00-\u9fff]", corpus[llm1][key.split("@@@")[1]]["text"]) or re.search("[\u4e00-\u9fff]", corpus[llm2][key.split("@@@")[1]]["text"]):
                    continue
                if args.select_by == 'rank':
                    diff1 = avg_rank_diff_dict[llm1][key]
                    diff2 = avg_rank_diff_dict[llm2][key]
                elif args.select_by == 'score':
                    diff1 = avg_score_diff_dict[llm1][key]
                    diff2 = avg_score_diff_dict[llm2][key]
                else:
                    continue  # invalid select_by option

                if diff1 is None or diff2 is None:
                    continue  # Skip if any of the diffs is None
                
                # calculate similarity
                human_text = corpus["human"][key.split("@@@")[1]]["text"]
                llm1_text = corpus[llm1][key.split("@@@")[1]]["text"]
                llm2_text = corpus[llm2][key.split("@@@")[1]]["text"]

                if args.select_by == 'rank' and diff1 > args.threshold and diff2 < -args.threshold:
                    count += 1
                    prefer_diff = diff1 - diff2
                    max_diff = max(max_diff, prefer_diff)
                    min_diff = min(min_diff, prefer_diff)
                    preference_samples.append([human_text, llm1_text, llm2_text, prefer_diff])
                elif args.select_by == 'score' and diff1 < -args.threshold and diff2 > args.threshold:
                    count += 1
                    prefer_diff = diff2 - diff1
                    max_diff = max(max_diff, prefer_diff)
                    min_diff = min(min_diff, prefer_diff)
                    preference_samples.append([human_text, llm1_text, llm2_text, prefer_diff])

    print(f"Total count of selected documents: {count}")
    print(f"Max prefer_diff: {max_diff}")
    print(f"Min prefer_diff: {min_diff}")
    
    filtered_preference_samples = []
    max_diff = float('-inf')
    min_diff = float('inf')
    for human_text, llm1_text, llm2_text, prefer_diff in preference_samples:
        sim_llm1_human = compute_similarity(bge_model, llm1_text, human_text)
        sim_llm2_human = compute_similarity(bge_model, llm2_text, human_text)
        max_diff = max(max_diff, prefer_diff)
        min_diff = min(min_diff, prefer_diff)
        
        if sim_llm1_human >= 0.92 and sim_llm2_human >= 0.92:
            filtered_preference_samples.append([human_text, llm1_text, llm2_text, prefer_diff])

    print(f"Total count of filtered documents: {len(filtered_preference_samples)}")
    print(f"Max prefer_diff: {max_diff}")
    print(f"Min prefer_diff: {min_diff}")

    full_preference_samples = []
    for prompt_id, prompt_data in zip(args.prompt_id_list, prompt_data_list):
        template_preference_samples = []
        
        for human_text, llm1_text, llm2_text, prefer_diff in filtered_preference_samples:
            template_preference_samples = apply_template(prompt_data, template_preference_samples, human_text, llm1_text, llm2_text, prefer_diff)
            
        # Normalize prefer_diff values
        template_preference_samples = normalize_prefer_diff(template_preference_samples, min_diff, max_diff)
        full_preference_samples.extend(template_preference_samples)

    # save all the preference samples
    with open(f"{args.output_path}/dpo-{args.select_by}{args.threshold}.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(full_preference_samples, ensure_ascii=False, indent=4) + '\n')

    logging.info("==========Done!==========")


if __name__ == "__main__":
    main()

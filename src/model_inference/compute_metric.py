import argparse
import json
import os
import re
import random
import datasets

from pathlib import Path

cwd = Path(__file__).parent.absolute()

def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1


if __name__ == "__main__":
    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    
    options = ["A", "B", "C", "D", "E"]
    dataset = datasets.load_dataset("cnut1648/ScienceQA-LLAVA")['test']
    with open(cwd / "scienceqa-eval.jsonl") as f:
        predictions = [
            json.loads(line) for line in f.readlines()
        ]
    assert len(dataset) == len(predictions)
    for prob, pred in zip(dataset, predictions):
        pred_text = pred['text']
        prob_id = pred['question_id']

        if pred_text in options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in options and pred_text[1:3] == ". ":
            answer = pred_text[0]
        else:
            pattern = re.compile(r'The answer is ([A-Z]).')
            res = pattern.findall(pred_text)
            if len(res) == 1:
                answer = res[0]  # 'A', 'B', ...
            else:
                answer = "FAILED"

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': prob['answer'],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        if answer == prob['answer']:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    ###### IMG ######
    multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    multimodal_total = multimodal_correct + multimodal_incorrect
    ###### IMG ######

    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    with open(cwd/"scienceqa-eval-result.json", 'w') as f:
        json.dump(sqa_results, f, indent=2)

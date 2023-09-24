import datasets
import pandas
from tqdm.auto import tqdm

def convert_to_llava(split: str):# base_dir, split, prompt_format="QCM-LEA"):
    dataset = datasets.load_dataset('derek-thomas/ScienceQA')[split].cast_column("image", datasets.Image(decode=False))
    target_format = []
    for i, instance in tqdm(enumerate(dataset), total=len(dataset)):
        answer_str = instance['choices'][instance['answer']]
        metadata = {
            'solution': instance['solution'], # step by step solution
            "lecture": instance['lecture'], # lecture name
            "topic": instance['topic'], "subject": instance['subject'], "category": instance['category'], "grade": instance['grade'],
        }
        if instance['image'] is None: # no image
            target_format.append({
                "id": f"{split}-{i}",
                "conversations": [
                    {'from': 'human', 'value': instance['question']},
                    {'from': 'gpt', 'value': answer_str},
                ],
                **metadata
            })
        else:
            target_format.append({
                "id": f"{split}-{i}",
                "image": instance['image'],
                "conversations": [
                    {'from': 'human', 'value': instance['question'] + '\n<image>'},
                    {'from': 'gpt', 'value': answer_str},
                ],
                **metadata
            })
    target_format = datasets.Dataset.from_pandas(pandas.DataFrame(target_format)).cast_column("image", datasets.Image(decode=True))
    print(f'[{split}] Number of samples: {len(target_format)}')
    return target_format


if __name__ == "__main__":
    orig_dataset = datasets.load_dataset('derek-thomas/ScienceQA')
    orig_dataset.save_to_disk("ScienceQA")
    formatted_dataset = {}
    for split in ["train", "validation", "test"]:
        formatted_dataset[split] = convert_to_llava(split)
    formatted_dataset = datasets.DatasetDict(formatted_dataset)
    formatted_dataset.save_to_disk("ScienceQA-LLAVA")

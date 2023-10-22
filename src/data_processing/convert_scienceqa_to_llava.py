import datasets
import pandas
from tqdm.auto import tqdm
from utils import get_question_text, get_context_text, get_choice_text, get_answer, get_lecture_text, get_solution_text, create_one_example_chatbot

def convert_to_llava(split: str):# base_dir, split, prompt_format="QCM-LEA"):
    dataset = datasets.load_dataset('derek-thomas/ScienceQA')[split].cast_column("image", datasets.Image(decode=False))
    target_format = []
    use_caption=False
    prompt_format = "CQM-A"
    is_test = False
    options=["A", "B", "C", "D", "E"]

    for i, instance in tqdm(enumerate(dataset), total=len(dataset)):
        question = get_question_text(instance)
        context = get_context_text(instance, use_caption)
        choice = get_choice_text(instance, options)
        answer = get_answer(instance, options)
        lecture = get_lecture_text(instance).replace('\\n', '\n')
        solution = get_solution_text(instance).replace('\\n', '\n')
        
        input, output = create_one_example_chatbot(prompt_format, question, context, choice, answer, lecture, solution, test_example=is_test)
        
        if input.startswith('Question: '):
            input = input.replace('Question: ', '')
        if output.startswith('Answer: '):
            output = output.replace('Answer: ', '')
        
        metadata = {
            "question": question,
            "context": context,
            "choice": choice,
            "answer": answer,
            "lecture": lecture,
            "solution": solution,
        }

        if instance['image'] is None: # no image
            target_format.append({
                "id": f"{split}-{i}",
                "conversations": [
                    {'from': 'human', 'value': input},
                    {'from': 'gpt', 'value': output},
                ],
                **metadata
            })
        else:
            target_format.append({
                "id": f"{split}-{i}",
                "image": instance['image'],
                "conversations": [
                    {'from': 'human', 'value': f"{input}\n<image>"},
                    {'from': 'gpt', 'value': output},
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

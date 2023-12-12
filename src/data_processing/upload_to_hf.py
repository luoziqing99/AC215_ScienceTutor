import datasets
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")

dataset = datasets.load_from_disk("ScienceQA-LLAVA")
dataset.push_to_hub("cnut1648/ScienceQA-LLAVA", private=True, token=token)
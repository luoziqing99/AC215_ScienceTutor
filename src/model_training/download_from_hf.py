import datasets
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HF_TOKEN")
ds = datasets.load_dataset("cnut1648/ScienceQA-LLAVA", token=token)
ds.save_to_disk("ScienceQA-LLAVA")
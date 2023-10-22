from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "wandb",
    "huggingface_hub",
    "transformers",
    "datasets",
    "evaluate",
    # "fire",
    # "flash_attn",
    # "deepspeed",
    # "accelerate"
]

setup(
    name="ac215-sciencetutor-trainer",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="ScienceTutor Trainer Application",
)

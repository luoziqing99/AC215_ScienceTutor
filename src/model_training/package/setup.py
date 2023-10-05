from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "flash_attn",
    "transformers",
    "datasets",
    "evaluate",
    "fire",
    "deepspeed",
    "accelerate"
]

setup(
    name="ac215-sciencetutor-trainer",
    version="0.0.1",
    # install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="ScienceTutor Trainer Application",
)

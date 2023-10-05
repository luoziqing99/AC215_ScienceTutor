import subprocess
import os
# from dotenv import load_dotenv

# load_dotenv()

script = """\
git clone https://github.com/cnut1648/LLaVA
pip install transformers datasets evaluate
pip install ninja
pip install flash-attn --no-build-isolation
pip install fire
cd LLaVA
pip install -e .
mkdir checkpoints
cd checkpoints
wget "https://huggingface.co/liuhaotian/llava-pretrain-vicuna-7b-v1.3/resolve/main/mm_projector.bin"
cd ..

# Weights and Biases
wandb login "7c8a8658dd63a3d0259cf220ea3482c9e36431e5"

pip list

pip install flash-attn --no-build-isolation

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --bits 4 \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --version v1 \
    --data_path ../ScienceQA/data/scienceqa/llava_train_QCM-LEA.json \
    --image_folder ../ScienceQA/data/scienceqa/images/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./checkpoints/llava-vicuna-7b-v1.3-pretrain-ScienceQA_QCM_LEA \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
"""

with open(os.path.join(os.getcwd(), "task-shell.sh"), "w") as f:
    f.write(script)
os.chmod("task-shell.sh", 0o777)
# Define the shell script command as a string
shell_script_command = "./task-shell.sh"

# Use subprocess to run the shell script
process = subprocess.Popen(shell_script_command, shell=True)

# Wait for the shell script to complete
process.wait()

# Check the return code
if process.returncode == 0:
    print("Shell script executed successfully")
else:
    print(f"Shell script failed with return code {process.returncode}")
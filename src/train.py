from datetime import datetime
import secrets
import modal
import os
import requests
from .kmer import kmer_featurization

from .common import (
    stub,
    axolotl_image,
    VOLUME_CONFIG,
)

N_GPUS = int(os.environ.get("N_GPUS", 2))
GPU_MEM = int(os.environ.get("GPU_MEM", 80))
GPU_CONFIG = modal.gpu.A100(count=N_GPUS, memory=GPU_MEM)


def print_common_training_issues(config):
    min_train_tokens = (
            config["sequence_len"]
            * config["gradient_accumulation_steps"]
            * config["micro_batch_size"]
            * N_GPUS
    )
    print(
        f"Please ensure there are enough tokens to train a single epoch of {min_train_tokens} tokens (recommended to have 4x)."
    )

    min_eval_samples = config["micro_batch_size"] * N_GPUS
    print(
        f"Please ensure there are enough samples for evaluation ({min_eval_samples})."
    )


def run_cmd(cmd: str, run_folder: str):
    import subprocess

    # Ensure volumes contain latest files.
    VOLUME_CONFIG["/pretrained"].reload()
    VOLUME_CONFIG["/runs"].reload()

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):
        exit(exit_code)

    # Commit writes to volume.
    VOLUME_CONFIG["/runs"].commit()


@stub.function(
    image=axolotl_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=3600 * 24,
    _allow_background_volume_commits=True,
)
def train(run_folder: str, merge_only=False):
    print(f"Starting training run in {run_folder}")

    if not merge_only:
        TRAIN_CMD = "accelerate launch -m axolotl.cli.train ./config.yml"
        run_cmd(TRAIN_CMD, run_folder)

    # Kick off CPU job to merge the LoRA weights into base model.
    merge_handle = merge.spawn(run_folder)
    with open(f"{run_folder}/logs.txt", "a") as f:
        f.write(f"<br>merge: https://modal.com/logs/call/{merge_handle.object_id}\n")
        print(f"Beginning merge {merge_handle.object_id}.")
    return merge_handle


@stub.function(image=axolotl_image,
               volumes=VOLUME_CONFIG,
               timeout=3600 * 24)
def merge(run_folder: str):
    import glob
    import yaml
    import shutil

    shutil.rmtree(f"{run_folder}/lora-out/merged", ignore_errors=True)

    with open(f"{run_folder}/config.yml") as config:
        # Loading ./lora-out saved by deepspeed has issues, use latest checkpoint instead.
        if yaml.safe_load(config).get("deepspeed", None):
            checkpoints = glob.glob(f"./lora-out/checkpoint-*", root_dir=run_folder)
            MERGE_SRC = max(checkpoints, key=lambda path: int(path.split("-")[-1]))
        else:
            MERGE_SRC = "./lora-out"

        print(f"Merge from {MERGE_SRC} in {run_folder}")

    MERGE_CMD = f"accelerate launch -m axolotl.cli.merge_lora ./config.yml --lora_model_dir='{MERGE_SRC}' --load_in_8bit=False --load_in_4bit=False --flash_attention=False"
    run_cmd(MERGE_CMD, run_folder)

    VOLUME_CONFIG["/runs"].commit()


@stub.function(
    image=axolotl_image,
    timeout=60 * 30,
    volumes=VOLUME_CONFIG,
    # mounts=modal.Mount.from_local_python_packages("kmer")
)
def launch(config_raw: str, use_kmer: bool = True):
    from huggingface_hub import snapshot_download
    import yaml
    import json, jsonlines

    # Ensure the base model is downloaded
    # TODO(gongy): test if this works with a path to previous fine-tune
    config = yaml.safe_load(config_raw)
    model_name = config["base_model"]
    dataset_url = config['dataset_url']

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} ...")
        snapshot_download(model_name)

        print("Committing /pretrained directory (no progress bar) ...")
        VOLUME_CONFIG["/pretrained"].commit()

    # Write config and data into a training subfolder.
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_folder = f"/runs/axo-{time_string}-{secrets.token_hex(2)}"
    os.makedirs(run_folder)

    print(f"Preparing training run in {run_folder}.")
    temp_file_name = "my_data.jsonl"
    data_file_name = config['datasets'][0]['path']
    print("Downloading data...")

    r = requests.get(dataset_url)
    data_raw = r.text
    print(data_raw[:20])

    if not use_kmer:
        temp_file_name = data_file_name

    with open(f"{run_folder}/{temp_file_name}", "w") as data_file:
        print("Writing data...")
        data_file.write(data_raw)

    if use_kmer:
        print("Using kmer...")
        featurizer = kmer_featurization(5)
        with open(f"{run_folder}/{temp_file_name}", 'r') as r, jsonlines.open(f"{run_folder}/{data_file_name}", 'w') as w:
            for line in r:
                record = json.loads(line)
                seq = record['context'].split(" ")
                kmer_feature = featurizer.obtain_kmer_feature_for_a_list_of_sequences(seq)

                record['question'] = record['question'].strip()
                record['context'] = kmer_feature
                record['answer'] = record['answer'].strip()

                w.write(record)

    print("Writing config...")
    with open(f"{run_folder}/config.yml", "w") as config_file:
        config_file.write(config_raw)

    VOLUME_CONFIG["/runs"].commit()

    # Start training run.
    train_handle = train.spawn(run_folder)
    with open(f"{run_folder}/logs.txt", "w") as f:
        f.write(f"train: https://modal.com/logs/call/{train_handle.object_id}")
    VOLUME_CONFIG["/runs"].commit()

    return run_folder, train_handle


@stub.local_entrypoint()
def main(config: str = "config.yml", use_kmer: bool = True):
    # Read config.yml and my_data.jsonl and pass them to the new function.
    print("Reading config...")
    dir = os.path.dirname(__file__)
    with open(f"{dir}/{config}", "r") as cfg:
        config_content = cfg.read()

    _, train_handle = launch.remote(config_content, use_kmer)

    # Wait for the training run to finish.
    merge_handle = train_handle.get()
    merge_handle.get()


# @stub.local_entrypoint()
# def main_merge(run_folder: str):
#     # Read config.yml and my_data.jsonl and pass them to the new function.
#     print("Starting Merge...")
#     train_handle = train.spawn(run_folder, True)
#
#     # Wait for the training run to finish.
#     merge_handle = train_handle.get()
#     merge_handle.get()

from modal import Secret
from datetime import datetime

from common import (
    WANDB_PROJECT,
    process_prompt,
    stub,
    VOLUMES,
    MODEL_NAME,
    GPU_CONFIG
)

from protllama import llama_train


@stub.function(
    gpu=GPU_CONFIG,
    cpu=20,
    # TODO: Modal should support optional secrets.
    secrets=[
        Secret.from_name("my-wandb-secret"),
        Secret.from_name("my-huggingface-secret"),
    ],
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    allow_cross_region_volumes=True,
)
def finetune(process_data=True):
    import jsonlines
    import pandas as pd
    from datasets import Dataset
    from huggingface_hub import snapshot_download

    from sklearn.model_selection import train_test_split

    VOLUMES['/vol'].reload()
    VOLUMES['/model'].reload()

    data = None
    if process_data:
        print('Loading JSON...')
        train = list(jsonlines.open('/vol/scop/results/soft_rh_train_new_dataset.json', 'r').iter())

        train = [process_prompt(raw) for raw in train]

        train = pd.DataFrame(train)

        train, test = train_test_split(train, train_size=150000, shuffle=True, stratify=train['label'])
        test, _ = train_test_split(test, train_size=50000, shuffle=True, stratify=test['label'])
        print(train.shape, test.shape)

        print('Loading Dataset...')
        train = Dataset.from_pandas(train)
        test = Dataset.from_pandas(test)

        print("Splitting dataset...")
        data = train.train_test_split(train_size=.85, seed=42)
        data['val'] = data.pop('test')
        data['test'] = test

        num_samples = len(data["train"])
        print(f"Loaded {num_samples} samples. ")

    try:
        snapshot_download(MODEL_NAME, local_files_only=True)
        print(f"Volume contains {MODEL_NAME}.")
    except FileNotFoundError:
        print(f"Downloading {MODEL_NAME} ...")
        snapshot_download(MODEL_NAME)

        print("Committing /pretrained directory (no progress bar) ...")
        VOLUMES["/model"].commit()

    print("Start trainer...")
    llama_train(
        data,
        output_dir="./prollama-7b-lora-8-remote-homology-filtered",
        batch_size=8,
        micro_batch_size=8,
        num_epochs=5,
        learning_rate=2e-4,
        lora_r=8,
        lora_alpha=8,
        lora_task_type='SEQ_CLS',
        lora_bias='none',
        lora_dropout=0.1,
        wandb_project=WANDB_PROJECT,
        cutoff_len=512,
        wandb_run_name=f"prollama-7b-lora-16-remote-homology-filtered-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    )

    VOLUMES["/model"].commit()

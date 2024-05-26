import modal
from modal import Image, App, Volume, gpu

WANDB_PROJECT = "huggingface"

MODEL_NAME = "GreatCaptainNemo/ProLLaMA"
MAX_LEN = 512

id2label = {0: "NON-HOMOLOGS", 1: "REMOTE-HOMOLOGS"}
label2id = {"NON-HOMOLOGS": 0, "REMOTE-HOMOLOGS": 1}

openllama_image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit",
        "cudnn",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "accelerate",
        "datasets",
        "peft",
        "transformers",
        "torch",
        "torchvision",
        "sentencepiece",
        "jsonlines",
        "evaluate",
        "huggingface_hub",
        "hf-transfer",
        "scikit-learn",
        "pandas"
    )
    # .run_function(download_models)
    .pip_install("wandb")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/model",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            # TQDM_DISABLE="true",
        )
    )
)

data_vol = Volume.from_name("homology-volume", create_if_missing=True)
model_vol = Volume.from_name("homology-model-volume", create_if_missing=True)

VOLUMES = {
    '/vol': data_vol,
    '/model': model_vol
}

stub = App(name="homology-finetune", image=openllama_image)

# prompt_gen = """
# {fa_query}
# [CLS]
# {fa_context}
# """
prompt_gen = """
[Determine Homology]
SeqPiFamily={fa_query}
SeqPjFamily={fa_context}
"""


def process_prompt(raw):
    label = 1 if raw['output'] else 0
    raw.pop('output')
    result = {"label": label, 'text': prompt_gen.format(**raw)}
    return result


def compute_metrics(eval_pred):
    import evaluate
    import numpy as np

    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred  # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


GPU_CONFIG = gpu.H100(count=3)

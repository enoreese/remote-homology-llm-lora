from .common import (
    stub,
    axolotl_image,
    VOLUME_CONFIG,
)


@stub.function(image=axolotl_image, timeout=60 * 30, volumes=VOLUME_CONFIG)
def upload_to_hugginface(run_folder: str, model_name: str):
    from transformers import AutoModel

    print("Loading model")
    model = AutoModel.from_pretrained(f'{run_folder}/lora-out/merged')
    print("Uploading to HF...")
    model.push_to_hub(model_name)


@stub.local_entrypoint()
def upload(run_folder: str, model_name: str):
    print("Uploading...")
    upload_to_hugginface.remote(run_folder, model_name)

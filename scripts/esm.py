from common import compute_metrics, MAX_LEN, id2label, label2id, MODEL_NAME, VOLUMES


def esm_train(
        # model/data params
        data,
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        cutoff_len: int = MAX_LEN,
        num_process=50,
        # lora hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
        lora_bias='none',
        lora_task_type='CAUSAL_LM',
        # llm hyperparams
        class_weights=None,
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
):
    import os
    import sys

    import torch
    import transformers
    from peft import (
        LoraConfig,
        get_peft_model,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from accelerate import Accelerator

    accelerator = Accelerator()

    gradient_accumulation_steps = batch_size // micro_batch_size

    if lora_target_modules is None:
        lora_target_modules = ["query", "key", "value"]

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id,
    )
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=cutoff_len)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=lora_task_type,
    )
    model = get_peft_model(model, config)

    print(model.print_trainable_parameters())  # Be more transparent about the % of trainable params.

    print("Tokenizing dataset...")
    if os.path.isdir('/vol/scop/results/tokenized_datasets') and data is None:
        from datasets import load_from_disk

        tokenized_datasets = load_from_disk("/vol/scop/results/tokenized_datasets")
        tokenized_datasets.set_format("torch")
    else:
        tokenized_datasets = data.map(tokenize, batched=True, batch_size=2500)
        tokenized_datasets.set_format("torch")

        print("Saving dataset...")
        tokenized_datasets.save_to_disk('/vol/scop/results/tokenized_datasets')
        VOLUMES["/vol"].commit()

    model = accelerator.prepare(model)
    tokenized_datasets = accelerator.prepare(tokenized_datasets)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=1.0,
            optim="adamw_torch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="wandb",
            fp16=True,
            gradient_checkpointing=True,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning starts...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(output_dir)

    trainer.push_to_hub()

    print("\n If there's a warning about missing keys above, please disregard :)")

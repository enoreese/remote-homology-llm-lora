from common import compute_metrics, MAX_LEN, id2label, label2id, MODEL_NAME, VOLUMES


def progen_train(
        # model/data params
        data,
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        cutoff_len: int = MAX_LEN,
        # lora hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
        lora_bias='none',
        lora_task_type='CAUSAL_LM',
        # llm hyperparams
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
):
    import os
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
    )
    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, AutoModelForCausalLM
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutputWithPast
    from typing import List, Optional, Tuple, Union
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
    from transformers.utils import (
        add_code_sample_docstrings,
        add_start_docstrings,
        add_start_docstrings_to_model_forward,
        is_flash_attn_2_available,
        is_flash_attn_greater_or_equal_2_10,
        logging,
        replace_return_docstrings,
    )

    if lora_target_modules is None:
        lora_target_modules = ["qkv_proj", "out_proj"]

    basemodel = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cuda",
                                                     trust_remote_code=True)

    class PGenPreTrainedModel(PreTrainedModel):
        config_class = basemodel.config_class
        base_model_prefix = "model"
        supports_gradient_checkpointing = True
        _skip_keys_device_placement = "past_key_values"
        _supports_flash_attn_2 = True
        _supports_cache_class = True

        def _init_weights(self, module):
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # custom class - modified from PhiForSequenceClassification
    class PGenForSequenceClassificationModified(PGenPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = 2  # changed
            self.model = basemodel.transformer  # changed
            self.score = nn.Linear(basemodel.config.hidden_size, self.num_labels, bias=False)  # changed

            # Initialize weights and apply final processing
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embd.wte  # changed

        def set_input_embeddings(self, value):
            self.model.embd.wte = value  # changed

        @add_start_docstrings_to_model_forward("PHI_INPUTS_DOCSTRING")
        def forward(
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            model_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            hidden_states = model_outputs.last_hidden_state  # changed
            logits = self.score(hidden_states)
            # print(logits)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                        logits.device
                    )
                else:
                    sequence_lengths = -1

            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(pooled_logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(pooled_logits, labels)
            if not return_dict:
                output = (pooled_logits,) + model_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )  # changed

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    basemodel.config.pad_token_id = tokenizer.pad_token_id
    model = PGenForSequenceClassificationModified(basemodel.config)

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

    print(model.print_trainable_parameters())

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

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="wandb",
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning starts...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(output_dir)

    trainer.push_to_hub()

    print("\n If there's a warning about missing keys above, please disregard :)")

# Protein Remote Homology Detection Using Large Language Models and LoRA

## Introduction

This repository contains research code and Jupyter notebooks for detecting remote homology in protein sequences using Protein Language Models (PLMs) such as:
- [ProGen 2](https://arxiv.org/pdf/2206.13517)
- [ProLLaMa](https://arxiv.org/pdf/2402.16445)
- [ESM](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)

Remote homology detection is a challenging task that involves identifying pairs of proteins with similar structures but low sequence similarity. Specifically, remote homology at the superfamily level involves proteins in the same superfamily but different families. Superfamily membership indicates similar structural characteristics, while family membership indicates high sequence similarity.

## Project Structure

The repository is organized as follows:

- **`notebooks/`**: Contains Jupyter notebooks used for data exploration, model training, and evaluation.
- **`scripts/`**: Contains Python scripts for preprocessing data, training models, and running experiments.
- **`README.md`**: Project overview and instructions.

## Requirements

To run the code in this repository, you need the following dependencies:

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.5+
- NumPy
- Pandas
- Scikit-learn
- HuggingFace Datasets
- Jupyter Notebook
- WandB (optional for experiment tracking)

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/enoreese/remote-homology-llm-lora.git
cd remote-homology-llm-lora
```

### 2. Prepare the Data

Download and preprocess the dataset using the provided scripts. Ensure that the data is placed in the `data/` directory.

```bash
python scripts/SCOP_processing.py
```

### 3. Fine-tune the Model

Train the model using the provided training script. You can customize the training parameters in the configuration file located in the `config/` directory.

```bash
modal scripts/finetune.py::finetune
```

### 4. Evaluate the Model

Evaluate the fine-tuned model on the validation dataset.

```bash
modal scripts/evaluate.py::evaluate
```

## Usage

### Fine-tuned Models

We provide fine-tuned models for remote homology detection. You can download and use these models for your research:

- [ESM-8M](https://huggingface.co/sasuface/esm2-t6-8M-lora-256-remote-homology-filtered)
- [ESM-35M](https://huggingface.co/sasuface/esm2-t12-35M-lora-64-remote-homology-filtered)
- [ESM-3B](https://huggingface.co/sasuface/esm2-t36-3B-lora-16-remote-homology-filtered)
- [ProGen 2](https://huggingface.co/sasuface/progen2-small-lora-64-remote-homology-filtered)
- [ProLLaMa](https://huggingface.co/sasuface/prollama-7b-lora-8-remote-homology-filtered)

You can load these models using the following code:

```python
from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

prompt = """
[Determine Homology]
SeqPiFamily=KADPCLTFNPDKCQLSFQPDGNRCAVLIKCGWECQSVAIQYKNKTRNNTLASTWQPGDPEWYTVSVPGADGFLRTVNNTFIFEHMCNTAMFMSRQYHMWPPRK
SeqPjFamily=QKLNLMQQTMSFLTHDLTQMMPRPVRGDQGQREPALLAGAGVLASESEGMRFVRGGVVNPLMRLPRSNLLTVGYRIHDGYLERLAWPLTDAAGSVKPTMQKLIPADSLRLQFYDGTRWQESWSSVQAIPVAVRMTLHSPQWGEIERIWLLRGPQ
"""

tokenizer = AutoTokenizer.from_pretrained('path/to/pretrained/model')
model = AutoModelForSequenceClassification.from_pretrained('path/to/pretrained/model')

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
prediction = pipe(prompt, return_all_scores=True)
```

## Contributing

We welcome contributions from the community. If you have suggestions or improvements, please open an issue or submit a pull request.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

We would like to thank the authors of PLMs and the HuggingFace Transformers library for their contributions to the open-source community. This research is built upon their work.

---

For any questions or issues, please contact [osas.usen@gmail.com].
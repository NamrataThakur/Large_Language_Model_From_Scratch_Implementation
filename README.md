# CustomGPT 🤖

CustomGPT is a re-implementation of 124 Million parameter GPT2 model **from scratch**, constructing each architectural component block-by-block using **PyTorch**, closely following the original design specifications. ✨

HuggingFace Repository: https://huggingface.co/NamrataThakur

## Overview 🔍

### VERSION 1 :

The model was trained using a transformer architecture with self-attention mechanisms to capture contextual relationships in text. In the version 1, we implemented three core fine-tuning strategies: **supervised classification fine-tuning (SFT)**, **instruction fine-tuning (IFT)** and **preference fine-tuning (PFT)** using **Direct Preference Optimization (DPO)** Loss. Each and every component is built from scratch.

Every component was built entirely **from scratch** with the goal of deeply understanding the inner workings of the so-called "black box" — the Large Language Model.

A fully modular training pipeline was also developed following Object-Oriented Programming (OOP) principles, enabling flexible integration of various training strategies and streamlined construction of custom data loaders.

## Model Architecture 🏗️

CustomGPT uses a standard 124M GPT decoder-only transformer architecture with the exact specifications mentioned in the paper: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- 12 transformer blocks 🧱
- 12 attention heads 👁️
- 768 embedding dimensions 📊
- Vocabulary size of 50,257 tokens 📚
- Context window of 1024 tokens 🪟

## Dataset 📖

The models (in version 1) were fine-tuned on the datasets provided in the book : [Build LLM From Scratch](https://sebastianraschka.com/books/)


## Models Trained:

- A classification model that can categorise spam vs ham messages.
- An instruct model (fine-tuned on GPT2 355M) to follow instructions on the Alpaca Prompt Template
- A preference-tuned model (on the instruct model) to give polite responses to the given queries.


## Installation 💿

To run the training pipeline for the CustomGPT, follow these steps:

```bash
# Clone the repository
git clone https://github.com/NamrataThakur/Large_Language_Model_From_Scratch_Implementation.git

#Create an environment:
python -m venv env

# Install the required packages
pip install -r requirements.txt

# Keep your training data in the "data" folder

# Change the training parameters according to the training type (SFT, IFT, PFT) in the run_gpt2_train.sh file

# Run the bash file containing training arguments
bash run_gpt2_train.sh
```

## Usage 🚀

### Chainlit Interface 🖥️

The easiest way to interact with fine-tuned models is through the Chainlit interface:

```bash
chainlit run main.py
```

This will launch a web application where you can input text and see the model's generated responses based on the type of model chosen: Chat Model or Classification Model. You can also change the text generation settings like Temperature, Top-K, and Max New Tokens for the Chat Model.

## Training ⚙️

The models were trained using PyTorch. The training process involved:

1. Tokenizing the input text
2. Creating sliding windows of fixed block size
3. Training the model with cross-entropy loss (for SFT and IFT) and direct preference optimization (DPO) loss (for PFT).
4. Applying learning rate scheduling with warmup and cosine decay.
5. Gradient clipping is applied to avoid too large gradient changes. The maximum norm threshold given is 1.


## Sample Outputs 📝

### Chat Model : 

- Chat Model (IFT / PFT) :

#### Example 1

```text
Prompt: What is the color of the sky?

Output:
The color of the sky is blue.
```

#### Example 2
  
```text
Prompt: What is the molecular formula for salt?

Output:
The molecular formula for salt is NaCl.

```

### Classification Model :

- Classification Model (SFT) :
  
#### Example 1

```
Input: 🔔 Amazon is hiring! Work from home and earn ₹5,000/day. No experience needed.
Register now: [joblink.xyz] – Limited slots!

Output:
Spam

```

#### Example 2

```
Input: Your SBI account will be blocked!
Update your KYC immediately at: [sbi-update-kyc.info]

Output:
Spam

```

## Inference 🔮

During inference, the models use several techniques to produce high-quality text:

- Temperature scaling for controlling randomness
- Top-k for focus and diversity
- Efficient token generation, one at a time, till the specified maximum token count is achieved

## Acknowledgement 🙏

We are deeply grateful to the following teachers who helped, taught, and inspired us to start this project:

- **Sebastian Raschka** for his excellent book [Build LLM From Scratch](https://sebastianraschka.com/books/)
- **Umar Jamil** for his deep [videos](https://www.youtube.com/@umarjamilai/videos) on topics like RLHF, DPO, LoRA etc.

## License 📜

This project is licensed under the MIT license - see the LICENSE file for details.

## Support ❤️

If you find CustomGPT useful, please consider starring the repository ⭐

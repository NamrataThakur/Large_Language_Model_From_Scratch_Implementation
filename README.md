# CustomGPT 🤖

CustomGPT is a re-implementation of 124 Million parameter GPT2 model **from scratch**, constructing each architectural component block-by-block using **PyTorch**, closely following the original design specifications. ✨

HuggingFace Repository: https://huggingface.co/NamrataThakur

## Overview 🔍

### VERSION 1 :

The model was trained using a transformer architecture with self-attention mechanisms to capture contextual relationships in text. In version 1, we implemented three core fine-tuning strategies: **supervised classification fine-tuning (SFT)**, **instruction fine-tuning (IFT)**, and **preference fine-tuning (PFT)** using **Direct Preference Optimization (DPO)** Loss. Each and every component is built from scratch.

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
- An instruct model (fine-tuned on GPT2 355M) to follow instructions using the Alpaca Prompt Template.
- A preference-tuned model (fine-tuned on the above instruct model) to give polite responses to the given queries.


## Installation 💿

To run the training pipeline, follow these steps:

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

# Clone the repository
git clone https://github.com/NamrataThakur/Large_Language_Model_From_Scratch_Implementation.git

#Create an environment:
python -m venv env

# Install the required packages
pip install -r requirements.txt

# Run the chainlit command to start the interface on localhost:
chainlit run main.py
```

This will launch a web application where you can input text and see the model's generated responses based on the type of model chosen: Chat Model or Classification Model. You can also change the text generation settings like Temperature, Top-K, and Max New Tokens for the Chat Model.


**Chainlit Demo Video:**

[![Chainlit Demo Video](https://img.youtube.com/vi/aePX4j1VrBk/0.jpg)](https://youtu.be/aePX4j1VrBk "Click to Play")



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


**Correct Responses:**
  
#### Example 1

```text
Prompt: definition of haughty

Output:
A haughty person is someone who is arrogant and conceited.
```

#### Example 2
  
```text
Prompt: What is the molecular formula for salt?

Output:
The molecular formula for salt is NaCl.

```

**Incorrect Responses:**

#### Example 1

```text
Prompt: Name the colors present in a rainbow?

Output:
The colors present in a rainbow are red, orange, yellow, and blue.
```

#### Example 2
  
```text
Prompt: opposite of haughty

Output:
Hughty is a type of arrogant.

```


### Classification Model :

- Classification Model (SFT) :

**Correct Responses:**
  
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


**Incorrect Responses:**

#### Example 1

```text
Prompt: Final notice: Your car insurance policy will be cancelled today. Renew instantly 👉

Output:
Ham
```

#### Example 2
  
```text
Prompt: Dear user, your electricity bill payment failed. Pay now to avoid disconnection:

Output:
Ham

```

The models presented in this version were fine-tuned using a relatively small dataset, which naturally limited their ability to generalize and, in turn, led to certain incorrect responses. However, achieving high predictive performance was not the primary objective at this stage. Instead, the central aim of Version 1 was to design and implement the complete training pipeline—covering **data preprocessing, model integration, fine-tuning strategies, and evaluation**—<u>from the ground up</u>. By focusing on building a fully functional, end-to-end pipeline, this version served as a foundational stage in which establishing modularity, scalability, and clarity of design took precedence over optimizing the performance of individual models.


<u>**In future iterations, our focus will shift toward systematically enhancing the accuracy of the models through refined training strategies and expanded datasets.</u>**


## Inference 🔮

During inference, the **instruction fine-tuned model** uses several techniques to produce high-quality text:

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

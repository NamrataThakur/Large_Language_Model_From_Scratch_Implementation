import torch
import os
os.pardir

from Instruction_finetuning.gpt2_instructDataFormat import format_input_response

def analyse_preferenceTuning(data, generate_policy, generate_reference, logger, n_records = None,
                             max_new_tokens = 256, temp = 0.0, top_k=None, eos_id = 50256):

    torch.manual_seed(123)
    
    if n_records is None:
        n_records = len(data)
    
    for row in data[:n_records]:
        _, input_text = format_input_response(row, inference=True)

        #generate = Text_Generation(model=policy_model, device=device, tokenizer_model='gpt2')
        output_text = generate_policy.text_generation(input_text = input_text, max_new_tokens=max_new_tokens, temp=temp, 
                                                      top_k= top_k, eos_id=eos_id)
        policy_output_text = (output_text[len(input_text) -2 :]).replace("### Response:", " ").replace('Response:', '').strip()

        #generate = Text_Generation(model=reference_model, device=device, tokenizer_model='gpt2')
        output_text = generate_reference.text_generation(input_text = input_text, max_new_tokens=max_new_tokens, temp= temp,
                                                         top_k= top_k, eos_id=eos_id)
        reference_output_text = (output_text[len(input_text) - 2:]).replace("### Response:", " ").replace('Response:', '').strip()

        logger.info(input_text)
        logger.info(f"\nCorrect response:\n>> {row['output']}")
        logger.info(f"\nReference model response:\n>> {reference_output_text.strip()}")
        logger.info(f"\nPolicy model response:\n>> {policy_output_text.strip()}")
        logger.info("\n-------------------------------------\n")

import torch

def save_model_response(data, generate, max_new_tokens = 256, temp = 0.0, top_k=None, eos_id = 50256):
    torch.manual_seed(123)

    for i, row in enumerate(data):
        _, input_text = format_input_response(row, inference=True)
        model_output = generate.text_generation(input_text = input_text, max_new_tokens=max_new_tokens, 
                                                temp=temp, top_k= top_k, eos_id = eos_id)
        model_response = model_output[len(input_text):].replace("### Response:", "").replace('Response:', '').strip()
        data[i]['model_response'] = model_response


    return data


def format_input_response(input_json, prompt_style = 'alpaca', inference=False):

    if prompt_style == 'alpaca':
        instruction = (f"Below is an instruction that describes a task. "
                       f"Write a response that appropriately completes the request."
                       f"\n\n### Instruction:\n {input_json['instruction']}"
                    )
        input = f"\n\n### Input:\n {input_json['input']}" if input_json['input'] else ""

        response = f"\n\n### Response:\n{input_json['output']}\n"

        formatted_input_with_response = instruction + input + response
        instruction_length = len(instruction + input)
        inf_format = instruction + input 

    else:

        instruction = f"\n\n<|user|>\n {input_json['instruction']} "
                            
        input = f": {input_json['input']}" if input_json['input'] else ""

        response =f"\n\n<|assistant|>\n {input_json['output']}"
                    

        formatted_input_with_response = instruction + input + response
        instruction_length = len(instruction + input)
        inf_format = instruction + input 

    return instruction_length, inf_format if inference else formatted_input_with_response 


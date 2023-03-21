import argparse
import re
import yaml
import gradio as gr
import torch
import os

### Model
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


TITLE = "Alpaca-LoRA Playground"

ABSTRACT = """
Thanks to [tolen](https://github.com/tloen/alpaca-lora), this simple application runs Alpaca-LoRA which is instruction fine-tuned version of [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) from Meta AI. Alpaca-LoRA is *Low-Rank LLaMA Instruct-Tuning* which is inspired by [Stanford Alpaca project](https://github.com/tatsu-lab/stanford_alpaca). Alpaca-LoRA is built on the same concept as Standford Alpaca project, but it was trained on a consumer GPU(RTX4090) with [transformers](https://huggingface.co/docs/transformers/index), [peft](https://github.com/huggingface/peft), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes/tree/main).
I am thankful to the [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU instances. 
"""

BOTTOM_LINE = """
In order to process batch generation, the common parameters in LLaMA are fixed. If you want to change the values of them, please do that in `generation_config.yaml`
"""

DEFAULT_EXAMPLES = [
    {
        "title": "1️⃣ List all Canadian provinces in alphabetical order.",
        "examples": [
            ["1", "List all Canadian provinces in alphabetical order."],
            ["2", "Which ones are on the east side?"],
            ["3", "What foods are famous in each province?"],
            ["4", "What about sightseeing? or landmarks?"],
        ],
    },
    {
        "title": "2️⃣ Tell me about Alpacas.",
        "examples": [
            ["1", "Tell me about alpacas."],
            ["2", "What other animals are living in the same area?"],
            ["3", "Are they the same species?"],
            ["4", "Write a Python program to return those species"],
        ],
    },
    {
        "title": "3️⃣ Tell me about the king of France in 2019.",
        "examples": [
            ["1", "Tell me about the king of France in 2019."],
        ]
    },
    {
        "title": "4️⃣ Write a Python program that prints the first 10 Fibonacci numbers.",
        "examples": [
            ["1", "Write a Python program that prints the first 10 Fibonacci numbers."],
            ["2", "could you explain how the code works?"]            
        ]
    }
]

SPECIAL_STRS = {
    "continue": "continue.",
    "summarize": "summarize our conversations so far in three sentences."
}

PARENT_BLOCK_CSS = """#col_container {width: 95%; margin-left: auto; margin-right: auto;}
#chatbot {height: 500px; overflow: auto;}
.chat_wrap_space {margin-left: 0.5em} """

def load_model(
    base="decapoda-research/llama-7b-hf",
    finetuned="tloen/alpaca-lora-7b"
):
    tokenizer = LlamaTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        base,
        load_in_8bit=True,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, finetuned, device_map={'': 0})
    return model,


def get_output(
    model, tokenizer, prompts, generation_config
):
    if len(prompts) == 1:
        print("there is only a prompt")
        encoding = tokenizer(prompts, return_tensors="pt")
        input_ids = encoding["input_ids"].cuda()
        generated_id = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_id)
        del input_ids, generated_id
        torch.cuda.empty_cache()
        return decoded
    else:
        print("there are multiple prompts")
        encodings = tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            **encodings,
            generation_config=generation_config,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_ids)
        del encodings, generated_ids
        torch.cuda.empty_cache()
        return decoded

def get_generation_config(path):
    with open('generation_config.yaml', 'rb') as f:
        generation_config = yaml.safe_load(f.read())

    return GenerationConfig(**generation_config["generation_config"])

def generate_prompt(prompt, histories, ctx=None):
    print("----inside")
    print(f"ctx:{ctx}")

    ctx = "" if ctx is None or ctx == "" else f"""
    
    Context:{ctx
    
    }"""

    convs = f"""Below is a history of instructions that describe tasks, paired with an input that provides further context. Write a response that appropriately completes the request by remembering the conversation history.
"""
    start_idx = 0
    
    for idx, history in enumerate(histories):
        history_prompt = history[0]
        if history_prompt == SPECIAL_STRS["summarize"]:
            start_idx = idx

    # drop the previous conversations if user has summarized
    for history in histories[start_idx if start_idx == 0 else start_idx+1:]:
        history_prompt = history[0]
        history_response = history[1]
        
        history_response = history_response.replace("<br>", "\n")

        pattern = re.compile(r'<.*?>')
        history_response = re.sub(pattern, '', history_response)        

        convs = convs + f"""### Instruction:{history_prompt}
### Response:{history_response}
"""

    convs = convs + f"""### Instruction:{prompt}
### Response:"""

    print(convs)
    return convs

def post_process(bot_response):
    bot_response = bot_response.split("### Response:")[-1].strip()
    bot_response = bot_response.replace("\n", "<br>")     # .replace(" ", "&nbsp;")
    
    pattern = r"(  )"
    replacement = r'<span class="chat_wrap_space">  <span>'
    return re.sub(pattern, replacement, bot_response)

def post_processes(bot_responses):
    return [post_process(r) for r in bot_responses]

def chat(
    contexts,
    instructions, 
    state_chatbots
):
    print("-------state_chatbots------")
    print(state_chatbots)
    results = []

    instruct_prompts = [
        generate_prompt(instruct, histories, ctx) 
        for ctx, instruct, histories in zip(contexts, instructions, state_chatbots)
    ]
        
    bot_responses = get_output(
        model, tokenizer, instruct_prompts, generation_config
    )
    print(bot_responses)
    bot_responses = post_processes(bot_responses)

    print("zipping...")
    sub_results = []
    for instruction, bot_response, state_chatbot in zip(instructions, bot_responses, state_chatbots):
        print(instruction)
        print(bot_response)
        print(state_chatbot)
        new_state_chatbot = state_chatbot + [(instruction, bot_response)]
        print(new_state_chatbot)
        results.append(new_state_chatbot)

    print(results)
    print(len(results))

    return (results, results)

def reset_textbox():
    return gr.Textbox.update(value='')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio Application for Alpaca-LoRA as a chatbot service"
    )
    # Dataset related.
    parser.add_argument(
        "--base_url",
        help="huggingface hub url",
        default="decapoda-research/llama-7b-hf",
        type=str,
    )
    parser.add_argument(
        "--ft_ckpt_url",
        help="huggingface hub url",
        default="tloen/alpaca-lora-7b",
        type=str,
    )
    parser.add_argument(
        "--port",
        help="port to serve app",
        default=os.environ.get("GRADIO_SERVER_PORT", 8080),
        type=int,
    )
    parser.add_argument(
        "--api_open",
        help="do you want to open as API",
        default="no",
        type=str,
    )
    parser.add_argument(
        "--share",
        help="do you want to share temporarily",
        default="no",
        type=str
    )

    return parser.parse_args()

def run(args):
    global model, tokenizer, generation_config

    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url
    )
    
    generation_config = get_generation_config()
    
    with gr.Blocks(css=PARENT_BLOCK_CSS) as demo:
        state_chatbot = gr.State([])

        with gr.Column(elem_id='col_container'):
            gr.Markdown(f"## {TITLE}\n\n\n{ABSTRACT}")

            with gr.Accordion("Context Setting", open=False):
                context_txtbox = gr.Textbox(placeholder="Surrounding information to AI", label="Enter Context")
                hidden_txtbox = gr.Textbox(placeholder="", label="Order", visible=False)

            chatbot = gr.Chatbot(elem_id='chatbot', label="Alpaca-LoRA")
            instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Instruction")
            send_prompt_btn = gr.Button(value="Send Prompt")

            gr.Markdown("#### Examples")
            for idx, examples in enumerate(DEFAULT_EXAMPLES):
                with gr.Accordion(examples["title"], open=False):
                    gr.Examples(
                        examples=examples["examples"], 
                        inputs=[
                            hidden_txtbox, instruction_txtbox
                        ],
                        label=None
                    )

            gr.Markdown(f"{BOTTOM_LINE}")

        send_prompt_btn.click(
            chat, 
            [context_txtbox, instruction_txtbox, state_chatbot],
            [state_chatbot, chatbot],
            batch=True,
            max_batch_size=4,
            api_name="text_gen"
        )
        send_prompt_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )

    demo.queue(
        api_open=False if args.api_open == "no" else True
    ).launch(
        share=False if args.share == "no" else True,
        server_port=args.port
    )

if __name__ == "__main__":
    args = parse_args()
    run(args)

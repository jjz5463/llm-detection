from datasets import load_dataset
from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import (
    DataFromPrompt,
    concat
)

# Load the dataset from Hugging Face
dataset = load_dataset("ChristophSchuhmann/essays-with-instructions")
# Extract the 'generated sentences' column
prompts = dataset['train']['instructions']

instruction = 'Write an essay with respect to following instruction: '

api = ''

with (((DataDreamer("./output")))):

    gpt_4 = OpenAI(model_name="gpt-4", api_key=api)

    datasets = []

    for i, prompt in enumerate(prompts):
        dataset = DataFromPrompt(
            f"llm detection sentence generation test part",
            args={
                "llm": gpt_4,
                "n": 1,
                "temperature": 1.0,
                "instruction": (
                        instruction + prompt + " Do not include title, the essay should have multiple paragraphs."
                ),
            },
            outputs={
                "prompts": "prompts",
                "generations": "generations by the LLM."
            },
        )
        datasets.append(dataset)

    probing_dataset = concat(*datasets, name='llm detection sentence generation usecase2 test')

    # Publish and share the synthetic dataset
    probing_dataset.publish_to_hf_hub(
        "jjz5463/llm-detection-generation-contribution2-test",
        token='',
        #private=True
    )
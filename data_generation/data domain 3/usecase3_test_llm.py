from datasets import load_dataset
from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import (
    DataFromPrompt,
    DataFromAttributedPrompt,
    concat
)
import pandas as pd

style = ['Expository', 'Descriptive', t, 'Narrative', 'Creative', 'Argumentative']
tone = ['excited', 'depressed', 'sarcastic', 'frightened', 'hopeful']
language = ['academic', 'formal', 'informal', 'Colloquial ']

# Load the dataset from Hugging Face
human_text = load_dataset("ChristophSchuhmann/essays-with-instructions")
# Extract the 'generated sentences' column
prompts = human_text['train']['instructions']

total_prompts_length = len(prompts)

instruction = (
    '''
    Here are some sample pieces of text written by high school students: 
        (1) {example1}
        (2) {example2}
        
    Now, write an essay simulates a high school student's writing
    using {style} writing styles, {tone} tone, {language} language, around length of {length} words
    with respect to following instruction:
    '''
)
#

api = ''

with (((DataDreamer("./output")))):

    gpt_4 = OpenAI(model_name="gpt-4", api_key=api)

    datasets = []

    for i, prompt in enumerate(prompts):

        length_essay = len(human_text['train'][i]['essays'].split())

        dataset = (
            DataFromAttributedPrompt(
                f"llm detection sentence generation fail test part",
                args={
                    "llm": gpt_4,
                    "n": 1,
                    "temperature": 1,
                    "top_p": 1,
                    "instruction": (
                        instruction + prompt +
                        " Do not include title, the essay should have multiple paragraphs."
                    ),
                    "attributes": {
                        'example1': [human_text['train'][i]['essays'][:10000]],
                        'example2': [human_text['train'][(i+1) % total_prompts_length]['essays'][:10000]],
                        'style': style,
                        'tone': tone,
                        'language': language,
                        'length': [length_essay]
                    }
                },
                outputs={
                    "prompts": "prompts",
                    "attributes": "attributes",
                    "generations": "generations by the LLM."
                },
            )
            .select_columns(["prompts", "attributes", "generations by the LLM."])
        )
        datasets.append(dataset)

    probing_dataset = concat(*datasets, name='llm detection sentence generation fail test case')

    # Publish and share the synthetic dataset
    probing_dataset.publish_to_hf_hub(
        "jjz5463/llm-detection-generation-failcase-test",
        token='',
        #private=True
    )
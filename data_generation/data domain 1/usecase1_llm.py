from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import (
    DataFromPrompt,
    concat
)
import pandas as pd

prompts_df = pd.read_csv("domain1_prompts.csv")

api = ''

with (((DataDreamer("./output")))):

    gpt_4 = OpenAI(model_name="gpt-4", api_key=api)

    prompts = [
        prompts_df['instructions'][0] + ' The followings are source text: ' + prompts_df['source_text'][0],
        prompts_df['instructions'][1] + ' The followings are source text: ' + prompts_df['source_text'][1] +
        " Consider additional format instruction: Do not include greeting like dear senator or sincerely or your name."
    ]

    datasets = []

    for i, prompt in enumerate(prompts):
        dataset = DataFromPrompt(
            f"llm detection sentence generation part",
            args={
                "llm": gpt_4,
                "n": 1000,
                "temperature": 1.0,
                "instruction": (
                        prompt + " Do not include title, the essay should have multiple paragraphs."
                ),
            },
            outputs={
                "prompts": "prompts",
                "generations": "generations by the LLM."
            },
        )
        datasets.append(dataset)

    probing_dataset = concat(*datasets, name='llm detection sentence generation 1.0')

    # Publish and share the synthetic dataset
    probing_dataset.publish_to_hf_hub(
        "jjz5463/llm-detection-generation-1.0",
        token='',
        #private=True
    )
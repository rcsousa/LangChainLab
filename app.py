"""
This module sends a test completion job to OpenAI's text-davinci-003 engine to generate a tagline for an ice cream shop.
"""

import openai
import dotenv

"""
This module sends a test completion job to OpenAI's text-davinci-003 engine to generate a tagline for an ice cream shop.
"""

import openai
import dotenv

config = dotenv.dotenv_values(".env")
openai.api_key = config['AZURE_OPENAI_KEY']
openai.api_base = config['AZURE_OPENAI_ENDPOINT'] 
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 
deployment_name='text-davinci-003' 


def generate_tagline(prompt: str, max_tokens: int) -> str:
    """
    Sends a completion call to OpenAI's text-davinci-003 engine to generate a tagline for an ice cream shop.

    Args:
        prompt (str): The prompt to send to the API.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The generated tagline.
    """
    try:
        response = openai.Completion.create(engine=deployment_name, prompt=prompt, max_tokens=max_tokens)
        text = response.choices[0].text.replace('\n', '').replace(' .', '.').strip()
        return text
    except Exception as e:
        print(f"Error generating tagline: {e}")
        return ""


if __name__ == "__main__":
    # Generate a tagline and print it
    print('Generating a tagline for an ice cream shop...')
    tagline = generate_tagline(prompt='Write a tagline for an ice cream shop. ', max_tokens=10)
    if tagline:
        print(f'Tagline: {tagline}')
    else:
        print('Failed to generate tagline.')
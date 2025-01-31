import yaml
import os
import json
import re
import time
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

api_key = ''

def pred_openai(model_name, msg):
    tries = 0
    while tries < 5:
        tries += 1
        logger.debug(f"Attempt {tries} to call OpenAI API")
        try:
            headers = {
                'Authorization': f"Bearer {api_key}"
            }
            logger.debug(f"Making request to OpenAI API with model: {model_name}")
            resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                "model": model_name,
                "messages": msg,
                "temperature": 0.
            }, headers=headers, timeout=120)
            
            if resp.status_code != 200:
                logger.error(f"API request failed with status {resp.status_code}: {resp.text}")
                raise Exception(resp.text)
            
            resp = resp.json()
            logger.debug("Successfully received API response")
            break
        except KeyboardInterrupt as e:
            logger.error("Keyboard interrupt received")
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                logger.error(f"Context length exceeded: {str(e)}")
                raise e
            logger.warning(f"Error occurred: {str(e)}. Retrying...")
            time.sleep(1)
    else:
        logger.error("Max retries reached. Failed to get response.")
        return
    
    return resp["choices"][0]["message"]["content"]

def get_criteria():
    logger.debug("Generating evaluation criteria")
    cri = 'For this evaluation, you should primarily consider the following criteria:\n'
    for key, value in CRITERIA.items():
        cri += f'{key}: {value}\n'
    return cri

def get_user_template(input, prediction, reference, criteria):
    logger.debug("Generating user template")
    return USER_TEMPLATE.format(
        input=input,
        prediction=prediction,
        reference=reference,
        criteria=criteria
    )

if __name__ == '__main__':
    logger.info("Starting evaluation script")
    
    try:
        with open('config-eval.yaml', 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            logger.debug(f"Loaded configuration: {config}")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

    pred_dir = config['pred_dir']
    save_dir = config['save_dir']
    model_name = config['model']['model_name']
    model_provider = config['model']['model_provider']
    
    criteria = get_criteria()
    reference = config['prompt']['needle']
    input = config['prompt']['retrieval_question']

    if not os.path.exists(save_dir):
        logger.info(f"Creating save directory: {save_dir}")
        os.makedirs(save_dir)

    result_dict = {}

    for filename in os.listdir(pred_dir):
        if not filename.endswith('.txt'):
            continue
        
        logger.info(f"Processing file: {filename}")
        
        try:
            with open(f'{pred_dir}/{filename}', 'r') as f:
                data = f.read().strip()
                logger.debug(f"File content length: {len(data)}")

            prediction = data
            user_template = get_user_template(input, prediction, reference, criteria)

            if model_provider == 'OpenAI':
                msg = [{
                        "role": "system",
                        "content": SYSTEM_TEMPLATE
                    }, {
                        "role": "user",
                        "content": user_template
                    }
                ]
                result = pred_openai(model_name, msg)
                
            else:
                logger.error(f'Unsupported model provider: {model_provider}')
                raise NotImplementedError(f'Not implemented model provider: {model_provider}')
            
            pattern = r"\[\[(\d+)\]\]"
            match = re.search(pattern, result)
            score = int(match.group(1)) if match else None
            logger.debug(f"Extracted score: {score}")

            result_dict[filename.replace('.txt', '')] = {
                'prediction': prediction,
                'score': score
            }
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")

    logger.info("Saving results")
    try:
        output_file = f'{save_dir}/{model_provider}_{model_name}_eval.json'
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=4)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")

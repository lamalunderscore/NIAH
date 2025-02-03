import yaml
import os
import json
import re
import time
import requests
from collections import deque
from datetime import datetime, timedelta

api_key = os.environ["OPENAI_API_KEY"]

def get_criteria():
    cri = 'For this evaluation, you should primarily consider the following criteria:\n'
    for key, value in CRITERIA.items():
        cri += f'{key}: {value}\n'
    return cri

CRITERIA = {
    "accuracy": """
    Score 1: The answer is completely unrelated to the reference.
    Score 3: The answer has minor relevance but does not align with the reference.
    Score 5: The answer has moderate relevance but contains inaccuracies.
    Score 7: The answer aligns with the reference but has minor omissions.
    Score 10: The answer is completely accurate and aligns perfectly with the reference.
    Only respond with a numberical score
    """
}

USER_TEMPLATE = '''[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. {criteria}[Ground truth]\n{reference}\nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n[The End of Assistant\'s Answer]'''

SYSTEM_TEMPLATE = 'You are a helpful assistant.'

def get_user_template(input, prediction, reference, criteria):
    return USER_TEMPLATE.format(
        input=input,
        prediction=prediction,
        reference=reference,
        criteria=criteria
    )

class TokenBucket:
    def __init__(self, tokens_per_min):
        self.capacity = tokens_per_min
        self.tokens = tokens_per_min
        self.last_updated = datetime.now()
        self.token_requests = deque()  # Store token usage with timestamps
        
    def _refill(self):
        now = datetime.now()
        time_passed = (now - self.last_updated).total_seconds()
        self.tokens = min(self.capacity, self.tokens + (time_passed * self.capacity / 60))
        self.last_updated = now
        
        # Remove token requests older than 1 minute
        while self.token_requests and self.token_requests[0] < now - timedelta(minutes=1):
            self.token_requests.popleft()

    def consume(self, tokens):
        self._refill()
        
        # Check if we've used too many tokens in the last minute
        if len(self.token_requests) >= self.capacity:
            return False
            
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.token_requests.append(datetime.now())
            return True
        return False

class RateLimitedOpenAI:
    def __init__(self, api_key, tokens_per_min=10000):
        self.api_key = api_key
        self.token_bucket = TokenBucket(tokens_per_min)
        
    def estimate_tokens(self, messages):
        # Rough estimation of tokens (4 chars ≈ 1 token)
        return sum(len(str(msg.get('content', ''))) // 4 for msg in messages)
    
    def pred_openai(self, model_name, messages, max_retries=5, backoff_factor=2):
        estimated_tokens = self.estimate_tokens(messages)
        
        for attempt in range(max_retries):
            # Check if we can consume tokens
            while not self.token_bucket.consume(estimated_tokens):
                time.sleep(0.1)  # Wait 100ms before trying again
                
            try:
                headers = {
                    'Authorization': f"Bearer {self.api_key}"
                }
                
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.
                    },
                    headers=headers,
                    timeout=120
                )
                
                if resp.status_code == 429:  # Rate limit error
                    retry_after = int(resp.headers.get('Retry-After', backoff_factor ** attempt))
                    time.sleep(retry_after)
                    continue
                    
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Max retries reached. Last error: {str(e)}")
                    
                if "maximum context length" in str(e):
                    raise e
                    
                print(f"Error Occurs: \"{str(e)}\"        Retry {attempt + 1}/{max_retries} ...")
                time.sleep(backoff_factor ** attempt)
                
        raise Exception("Max retries reached without successful response")

# Update the main execution code

if __name__ == '__main__':
    with open('config-eval.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    pred_dir = config['pred_dir']
    save_dir = config['save_dir']
    model_name = config['model']['model_name']
    model_provider = config['model']['model_provider']
    criteria = get_criteria()
    reference = config['prompt']['needle']
    input = config['prompt']['retrieval_question']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_dict = {}
    openai_client = RateLimitedOpenAI(api_key)

    for filename in os.listdir(pred_dir):
        if not filename.endswith('.txt'):
            continue

        with open(f'{pred_dir}/{filename}', 'r') as f:
            data = f.read().strip()

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
            result = openai_client.pred_openai(model_name, msg)
            
        else:
            raise NotImplementedError(f'Not implemented model provider: {model_provider}')
        
        pattern = r"\[\[(\d+)\]\]"
        match = re.search(pattern, result)
        score = int(match.group(1)) if match else None

        result_dict[filename.replace('.txt', '')] = {
            'prediction': prediction,
            'score': score
        }

    with open(f'{save_dir}/{model_provider}_{model_name}_eval.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

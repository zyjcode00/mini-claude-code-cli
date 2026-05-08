import openai
print('openai version:', openai.__version__)

client = openai.OpenAI(
    api_key='sk-fLjMgw5LbJh1VfGkCa08FaD8719a4f56906c41Bf16A451A7',
    base_url='https://api.shubiaobiao.cn/v1'
)

resp = client.chat.completions.create(
    model='deepseek-reasoner',
    messages=[{'role': 'user', 'content': '1+1=?'}],
    max_tokens=100,
    stream=True
)

reasoning_content_chunks = []
content_chunks = []
for chunk in resp:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta:
        if hasattr(delta, 'reasoning_content'):
            if delta.reasoning_content:
                reasoning_content_chunks.append(delta.reasoning_content)
        if delta.content:
            content_chunks.append(delta.content)
        if delta.model_extra:
            print('model_extra found:', delta.model_extra)

print(f'content: {"".join(content_chunks)}')
print(f'reasoning_content chunks: {len(reasoning_content_chunks)}')
if reasoning_content_chunks:
    print(f'reasoning_content: {"".join(reasoning_content_chunks)[:200]}')
print('DONE')
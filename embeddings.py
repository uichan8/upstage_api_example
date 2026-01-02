from openai import OpenAI

client = OpenAI(
    api_key="up_*************************imhd",
    base_url="https://api.upstage.ai/v1"
)

response = client.embeddings.create(
    input="Solar embeddings are awesome",
    model="embedding-query"
)

print(response.data[0].embedding)

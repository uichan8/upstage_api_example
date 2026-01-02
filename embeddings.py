import os
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# Upstage AI 클라이언트 초기화
client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

# 데이터
# 질문 (짧은 검색어나 질문에 사용)
query = "인공지능이란 무엇인가?"

# 데이터 (긴 텍스트나 문서 구절에 사용)
passages = [
    "인공지능은 기계에 의한 인간 지능의 시뮬레이션입니다.",
    "머신러닝은 데이터로부터 학습할 수 있는 AI의 하위 분야입니다.",
    "오늘 날씨는 맑습니다."
]

# 검색 쿼리 임베딩 (embedding-query)
query_response = client.embeddings.create(
    input=query,
    model="embedding-query"  # 검색 쿼리용
)
query_embedding = query_response.data[0].embedding
print(f"쿼리 임베딩 차원: {len(query_embedding)}")
print()

# 문서 구절 임베딩 (embedding-passage)
passage_embeddings = []
for passage in passages:
    response = client.embeddings.create(
        input=passage,
        model="embedding-passage"  # 문서 구절용
    )
    passage_embeddings.append(response.data[0].embedding)

print(f"문서 임베딩 차원: {len(passage_embeddings[0])}")
print()

# 유사도 계산
# Upstage는 정규화된 벡터를 반환하므로 dot product로 코사인 유사도 계산 가능
print("쿼리와 문서 간 유사도 점수:")
for i, passage in enumerate(passages):
    similarity = np.dot(query_embedding, passage_embeddings[i]) # dot product로 코사인 유사도 계산
    print(f"문서 {i+1}: {similarity:.4f} - {passage}")

# 가장 유사한 구절 찾기
most_similar_idx = np.argmax([np.dot(query_embedding, emb) for emb in passage_embeddings])
print(f"\n가장 유사한 문서: {passages[most_similar_idx]}")

# 결과를 markdown 파일로 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = f"result/embeddings/embedding_result_{timestamp}.md"

# 유사도 점수 계산
similarity_scores = [(i+1, np.dot(query_embedding, passage_embeddings[i]), passage)
                     for i, passage in enumerate(passages)]

# Markdown 내용 생성
markdown_content = f"""# Embeddings 실행 결과

## 실행 정보
- **실행 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **모델**: embedding-query, embedding-passage

## 검색 쿼리
```
{query}
```

## 문서 목록
"""

for i, passage in enumerate(passages):
    markdown_content += f"{i+1}. {passage}\n"

markdown_content += f"""
## 임베딩 정보
- **쿼리 임베딩 차원**: {len(query_embedding)}
- **문서 임베딩 차원**: {len(passage_embeddings[0])}

## 유사도 점수
코사인 유사도 (dot product):

| 순위 | 문서 번호 | 유사도 점수 | 문서 내용 |
|------|----------|------------|----------|
"""

# 유사도 점수를 내림차순으로 정렬
sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
for rank, (doc_num, score, passage) in enumerate(sorted_scores, 1):
    markdown_content += f"| {rank} | 문서 {doc_num} | {score:.4f} | {passage} |\n"

markdown_content += f"""
## 결론
가장 유사한 문서는 **문서 {most_similar_idx + 1}**입니다:
> {passages[most_similar_idx]}

유사도 점수: **{similarity_scores[most_similar_idx][1]:.4f}**
"""

# 파일 저장
os.makedirs("result/embeddings", exist_ok=True)
with open(result_file, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"\n결과가 저장되었습니다: {result_file}")

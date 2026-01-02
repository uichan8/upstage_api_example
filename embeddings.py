import os
import json
import time
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

# 데이터 파일 로드
print("=" * 80)
print("데이터 파일 로드 중...")
print("=" * 80)

# data.json 로드
data_path = "data/embeddings/data_002.json"
with open(data_path, "r", encoding="utf-8") as f: # 여기 수정
    data = json.load(f)

# 카테고리별로 문서를 평탄화
passages = []
category_info = []  # (문서 인덱스, 카테고리명) 정보 저장
for category, docs in data["passages"].items():
    for doc in docs:
        passages.append(doc)
        category_info.append(category)

queries = data["queries"]

print(f"✓ {len(passages)}개 문서 로드 완료")
print(f"✓ {len(queries)}개 쿼리 로드 완료")
print(f"✓ 카테고리: {list(data['passages'].keys())}")
print()

print("=" * 80)
print("임베딩 벡터 생성 중...")
print("=" * 80)
print()

# 모든 문서에 대한 임베딩 생성 (한 번만)
print(f"총 {len(passages)}개 문서의 임베딩을 생성합니다...")
passage_embeddings = []
for i, passage in enumerate(passages):
    response = client.embeddings.create(
        input=passage,
        model="embedding-passage"
    )
    passage_embeddings.append(response.data[0].embedding)
    if (i + 1) % 5 == 0:
        print(f"  {i + 1}/{len(passages)} 완료")

print(f"✓ 문서 임베딩 완료 (차원: {len(passage_embeddings[0])})")
print()

# 각 쿼리에 대한 검색 수행
all_query_results = []

for query_idx, query in enumerate(queries):
    print("=" * 80)
    print(f"쿼리 {query_idx + 1}/{len(queries)}: {query}")
    print("=" * 80)

    # 쿼리 임베딩 생성 (API 요청 시간 측정)
    print("API 요청 시작...")
    start_time = time.time()

    query_response = client.embeddings.create(
        input=query,
        model="embedding-query"
    )

    end_time = time.time()
    api_time = end_time - start_time

    query_embedding = query_response.data[0].embedding
    print(f"✓ API 응답 완료 (소요 시간: {api_time:.3f}초)")
    print(f"쿼리 임베딩 차원: {len(query_embedding)}")

    # 유사도 계산
    similarities = []
    for i, passage_emb in enumerate(passage_embeddings):
        similarity = np.dot(query_embedding, passage_emb)
        similarities.append((i + 1, similarity, passages[i]))

    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 상위 5개 결과 출력
    print(f"\n상위 5개 관련 문서:")
    for rank, (doc_num, score, passage) in enumerate(similarities[:5], 1):
        print(f"  {rank}. [문서 {doc_num}] {score:.4f} - {passage[:50]}...")

    print()

    # 결과 저장
    all_query_results.append({
        'query': query,
        'query_embedding': query_embedding,
        'similarities': similarities,
        'api_time': api_time
    })

# 결과를 markdown 파일로 저장
print("=" * 80)
print("결과를 파일로 저장 중...")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("result/embeddings", exist_ok=True)

# 각 쿼리별로 별도의 파일 생성
for query_idx, result in enumerate(all_query_results):
    query = result['query']
    similarities = result['similarities']
    api_time = result['api_time']

    # 파일명 생성
    result_file = f"result/embeddings/query_{query_idx + 1}_{timestamp}.md"

    # Markdown 내용 생성 (간단하게)
    markdown_content = f"""# 질문: {query}

**API 응답 시간**: {api_time:.3f}초

## 유사도 결과

| 순위 | 유사도 | 문서 내용 |
|------|--------|----------|
"""

    # 모든 결과를 유사도 순으로 표시
    for rank, (doc_num, score, passage) in enumerate(similarities, 1):
        markdown_content += f"| {rank} | {score:.4f} | {passage} |\n"

    # 파일 저장
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✓ 쿼리 {query_idx + 1} 결과 저장: {result_file}")
print()
print("=" * 80)
print("모든 작업 완료!")
print("=" * 80)

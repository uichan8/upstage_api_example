import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Upstage 유틸리티 함수 import
from upstage.embeddings import (
    create_passage_embeddings,
    search_query
)

# 데이터 파일 경로
DATA_PATH = "data/embeddings/data_002.json"


def load_data(data_path):
    """데이터 파일 로드

    Args:
        data_path: JSON 데이터 파일 경로

    Returns:
        tuple: (passages, queries, category_info)
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 카테고리별로 문서를 평탄화
    passages = []
    category_info = []
    for category, docs in data["passages"].items():
        for doc in docs:
            passages.append(doc)
            category_info.append(category)

    queries = data["queries"]

    return passages, queries, category_info


def save_results(query_results, timestamp):
    """결과를 markdown 파일로 저장

    Args:
        query_results: 쿼리 결과 리스트
        timestamp: 타임스탬프 문자열

    Returns:
        list: 저장된 파일 경로 리스트
    """
    os.makedirs("result/embeddings", exist_ok=True)

    saved_files = []
    for query_idx, result in enumerate(query_results):
        query = result['query']
        similarities = result['similarities']

        # 파일명 생성
        result_file = f"result/embeddings/query_{query_idx + 1}_{timestamp}.md"

        # Markdown 내용 생성
        markdown_content = f"""# 질문: {query}

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

        saved_files.append(result_file)

    return saved_files


def main():
    """메인 함수"""
    # .env 파일에서 환경변수 로드
    load_dotenv()

    # Upstage AI 클라이언트 초기화
    client = OpenAI(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1"
    )

    # 1. 데이터 로드
    print("=" * 80)
    print("데이터 파일 로드 중...")
    print("=" * 80)
    passages, queries, category_info = load_data(DATA_PATH)
    print(f"✓ {len(passages)}개 문서 로드 완료")
    print(f"✓ {len(queries)}개 쿼리 로드 완료")
    print()

    # 2. 문서 임베딩 생성
    print("=" * 80)
    print("임베딩 벡터 생성 중...")
    print("=" * 80)
    print(f"총 {len(passages)}개 문서의 임베딩을 생성합니다...")
    passage_embeddings = create_passage_embeddings(client, passages)
    print(f"✓ 문서 임베딩 완료 (차원: {len(passage_embeddings[0])})")
    print()

    # 3. 각 쿼리에 대한 검색 수행
    all_query_results = []

    for query_idx, query in enumerate(queries):
        print("=" * 80)
        print(f"쿼리 {query_idx + 1}/{len(queries)}: {query}")
        print("=" * 80)

        result = search_query(client, query, passage_embeddings, passages)

        # 결과 출력
        print(f"\n상위 5개 관련 문서:")
        for rank, (doc_num, score, passage) in enumerate(result['similarities'][:5], 1):
            print(f"  {rank}. [문서 {doc_num}] {score:.4f} - {passage[:50]}...")
        print()

        all_query_results.append(result)

    # 4. 결과 저장
    print("=" * 80)
    print("결과를 파일로 저장 중...")
    print("=" * 80)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = save_results(all_query_results, timestamp)

    for idx, file_path in enumerate(saved_files):
        print(f"✓ 쿼리 {idx + 1} 결과 저장: {file_path}")

    print()
    print("=" * 80)
    print("모든 작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

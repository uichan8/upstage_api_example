import numpy as np


def create_passage_embeddings(client, passages, model="embedding-passage"):
    """문서 임베딩 생성

    Args:
        client: OpenAI 클라이언트
        passages: 문서 리스트
        model: 임베딩 모델명 (기본값: "embedding-passage")

    Returns:
        list: 임베딩 벡터 리스트
    """
    passage_embeddings = []

    for passage in passages:
        response = client.embeddings.create(
            input=passage,
            model=model
        )
        passage_embeddings.append(response.data[0].embedding)

    return passage_embeddings


def create_query_embedding(client, query, model="embedding-query"):
    """쿼리 임베딩 생성

    Args:
        client: OpenAI 클라이언트
        query: 쿼리 문자열
        model: 임베딩 모델명 (기본값: "embedding-query")

    Returns:
        list: 임베딩 벡터
    """
    response = client.embeddings.create(
        input=query,
        model=model
    )

    return response.data[0].embedding


def search_query(client, query, passage_embeddings, passages):
    """쿼리 임베딩 생성 및 유사도 검색

    Args:
        client: OpenAI 클라이언트
        query: 쿼리 문자열
        passage_embeddings: 문서 임베딩 벡터 리스트
        passages: 문서 리스트

    Returns:
        dict: {
            'query': 쿼리,
            'query_embedding': 쿼리 임베딩,
            'similarities': 유사도 결과
        }
    """
    # 쿼리 임베딩 생성
    query_embedding = create_query_embedding(client, query)

    # 유사도 계산 (코사인 유사도)
    similarities = []
    for i, passage_emb in enumerate(passage_embeddings):
        # Upstage는 정규화된 벡터를 반환하므로 dot product로 코사인 유사도 계산
        similarity = np.dot(query_embedding, passage_emb)
        similarities.append((i + 1, similarity, passages[i]))

    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    return {
        'query': query,
        'query_embedding': query_embedding,
        'similarities': similarities
    }

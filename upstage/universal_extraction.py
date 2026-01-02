"""Upstage Document Parsing 유틸리티 함수"""

import base64
import json
import requests


def encode_image_to_base64(image_path):
    """이미지 파일을 base64로 인코딩

    Args:
        image_path: 이미지 파일 경로

    Returns:
        str: base64 인코딩된 이미지 문자열
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_url_to_base64(image_url):
    """URL에서 이미지를 다운로드하여 base64로 인코딩

    Args:
        image_url: 이미지 URL

    Returns:
        str: base64 인코딩된 이미지 문자열
    """
    response = requests.get(image_url)
    response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
    return base64.b64encode(response.content).decode("utf-8")


def universal_extraction_from_img(client, image_source, schema, model="information-extract", enhanced=False):
    """이미지에서 스키마에 따라 구조화된 정보 추출

    Args:
        client: OpenAI 클라이언트
        image_source: 이미지 파일 경로 또는 URL
        schema: 추출할 데이터 구조를 정의한 JSON 스키마
        model: 사용할 모델명 (기본값: "information-extract")
        enhanced: True일 경우 "enhanced" 모드, False일 경우 "standard" 모드 사용 (기본값: False)

    Returns:
        dict: 추출된 구조화된 데이터
    """
    # URL인지 로컬 파일인지 자동 감지
    if image_source.startswith('http://') or image_source.startswith('https://'):
        base64_image = encode_url_to_base64(image_source)
    else:
        base64_image = encode_image_to_base64(image_source)

    # 스키마 구조 분석 및 response_format 생성
    # Upstage 공식 형식: {"type": "json_schema", "json_schema": {...}}
    if isinstance(schema, dict) and "type" in schema and schema.get("type") == "json_schema":
        # 이미 전체 response_format 구조를 가진 경우 (공식 형식)
        response_format = schema
    else:
        # schema 부분만 있는 경우, response_format으로 감싸기
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_schema",
                "schema": schema
            }
        }

    # extra_body 설정
    extra_body = {
        "confidence": True,
        "mode": "enhanced" if enhanced else "standard"
    }

    # API 요청
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/octet-stream;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        response_format=response_format,
        extra_body=extra_body
    )

    # 결과 JSON 파싱 및 반환
    return json.loads(response.choices[0].message.content)

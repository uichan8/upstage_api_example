import os
import sys
import json
import argparse
import shutil
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Upstage 유틸리티 함수 import
from upstage.universal_extraction import universal_extraction_from_img

# 설정
SCHEMA_PATH = "data/universal_extraction/schema.json"  # 스키마 JSON 파일 경로
# IMAGE_PATH = "data/universal_extraction/거래명세서_스캔.JPG"  # 기본 이미지 경로
IMAGE_PATH = "https://postfiles.pstatic.net/MjAyMzA4MzFfMTM3/MDAxNjkzNDM2NzUzNDA3.5qmUxbhsWFmrS4p-7glAEvddUoy3YTdPkwVh6g1EVpAg.bskI705vR_Z1kyuXvLdu-XX37wxdlw15VNWEtXtKXOsg.JPEG.90viavz5/5.jpg?type=w773"


def load_schema(schema_path):
    """스키마 파일 로드

    Args:
        schema_path: 스키마 JSON 파일 경로

    Returns:
        dict: JSON 스키마 객체
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(result, timestamp, image_path):
    """추출 결과를 JSON과 Markdown 파일로 저장

    Args:
        result: 추출된 데이터
        timestamp: 타임스탬프 문자열
        image_path: 원본 이미지 경로

    Returns:
        tuple: (json_path, markdown_path, image_save_path)
    """
    # 타임스탬프별로 폴더 생성
    result_dir = f"result/universal_extraction/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # 이미지 저장
    is_url = image_path.startswith('http://') or image_path.startswith('https://')

    if is_url:
        # URL에서 이미지 다운로드
        response = requests.get(image_path)
        response.raise_for_status()

        # 확장자 추출 (없으면 .jpg 사용)
        ext = os.path.splitext(image_path.split('?')[0])[-1] or '.jpg'
        image_save_path = f"{result_dir}/image{ext}"

        with open(image_save_path, "wb") as f:
            f.write(response.content)
    else:
        # 로컬 파일 복사
        ext = os.path.splitext(image_path)[-1]
        image_save_path = f"{result_dir}/image{ext}"
        shutil.copy2(image_path, image_save_path)

    # JSON 파일로 저장
    json_path = f"{result_dir}/result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Markdown 파일로 저장
    markdown_path = f"{result_dir}/result.md"
    markdown_content = f"""# 문서 정보 추출 결과

## 입력 정보
- **이미지 파일**: {image_path}
- **추출 시간**: {timestamp}

## 추출된 데이터

```json
{json.dumps(result, ensure_ascii=False, indent=2)}
```
"""

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return json_path, markdown_path, image_save_path


def main():
    """메인 함수"""
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description="Upstage Document Parsing - 이미지에서 구조화된 정보 추출"
    )
    parser.add_argument(
        "image",
        nargs='?',
        default=IMAGE_PATH,
        help=f"이미지 파일 경로 또는 URL (기본값: {IMAGE_PATH})"
    )
    parser.add_argument(
        "-s", "--schema",
        default=SCHEMA_PATH,
        help=f"스키마 JSON 파일 경로 (기본값: {SCHEMA_PATH})"
    )
    args = parser.parse_args()

    # .env 파일에서 환경변수 로드
    load_dotenv()

    # Upstage AI 클라이언트 초기화
    client = OpenAI(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/information-extraction"
    )

    # 스키마 로드
    if not os.path.exists(args.schema):
        return

    schema = load_schema(args.schema)

    # 이미지 소스 확인
    is_url = args.image.startswith('http://') or args.image.startswith('https://')
    if not is_url and not os.path.exists(args.image):
        return

    # 문서 정보 추출
    result = universal_extraction_from_img(client, args.image, schema)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, markdown_path, image_save_path = save_result(result, timestamp, args.image)


if __name__ == "__main__":
    main()

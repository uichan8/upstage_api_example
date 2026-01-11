# Upstage API Example

Upstage AI API를 사용한 예제 프로젝트입니다.

공식 사이트 : https://upstage.ai/
api 문서 : https://console.upstage.ai/docs/getting-started

## 설치

1. 가상 환경 생성 및 활성화:
```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (Mac/Linux)
source venv/bin/activate

# 가상 환경 활성화 (Windows)
# venv\Scripts\activate
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 환경변수 설정

1. `.env.example` 파일을 `.env`로 복사:
```bash
cp .env.example .env
```

2. `.env` 파일을 열어서 실제 API 키로 수정:
```
UPSTAGE_API_KEY=your_actual_api_key_here
```
## Universal Extraction
API 페이지: https://console.upstage.ai/docs/capabilities/document-parse
설명: https://uichan8.notion.site/information-extract?source=copy_link

## Embeddings
API 페이지 : https://console.upstage.ai/docs/capabilities/embed
설명 : https://uichan8.notion.site/upstage-solar-embedding?source=copy_link

# Upstage API Example

Upstage AI API를 사용한 예제 프로젝트입니다.

공식 사이트 : https://upstage.ai/
api 문서 : https://console.upstage.ai/docs/getting-started

## 설치

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
설명: 이미지에서 구조화된 정보를 추출합니다.

### 사용 방법

**로컬 이미지 파일:**
```bash
python universal_extraction_example.py path/to/image.jpg
```

**URL 이미지:**
```bash
python universal_extraction_example.py https://example.com/image.jpg
```

**스키마 타입 지정:**
```bash
python universal_extraction_example.py image.jpg --schema receipt
python universal_extraction_example.py https://example.com/image.jpg -s business_card
```

### 사용 가능한 스키마
- `simple`: 간단한 영수증 정보 (업체명, 총액) - 기본값
- `receipt`: 상세 영수증 정보 (품목, 수량, 가격 등)
- `business_card`: 명함 정보 (이름, 회사, 직책, 연락처 등)
- `invoice`: 청구서 정보 (청구 번호, 품목, 세금 등)

### 결과
- `result/extraction/` 폴더에 JSON과 Markdown 형식으로 저장됩니다.

### 도움말
```bash
python universal_extraction_example.py --help
```

## Embeddings
API 페이지 : https://console.upstage.ai/docs/capabilities/embed
설명 : https://uichan8.notion.site/upstage-solar-embedding?source=copy_link

# 데이터안심구역 경진대회 2026

W-SIGNAL(물안부)은 상수도 생활신호와 소득통계를 활용해 돌봄 담당자의 지역 검토와 지원 연결을 돕는 프로젝트입니다. 지역 신호율 분석, 검침군별 사용량 예측, 웹 프로토타입을 담았습니다.

## 코드 구조

```text
src/
  preprocess_center.py   센터 원자료 전처리
  explore.py             지역 연결과 통계 분석
  validate_models.py     지역 모형 교차검증
  experiment.py          모형 후보와 시계열 비교
  finalize_models.py     채택 모형과 예측값 산출
  reproduce.py           입력 비교와 성능 지표 계산
  audit_priority.py      지수 민감도·공간 품질 분석
  review_policy.py       고정 검토량의 지역 선택 비교
  fetch_public_data.py   공공자료 확보와 지역 연계
  build_demo.py          공개 시연용 합성 데이터 생성
notebooks/               센터 전처리 노트북
data/origin/            승인 집계 배치 경로
data/public/            행정구역 연결표와 경계
data/derived/           지역별 파생값 배치 경로
results/                 공공자료·모형 성능 요약
app/index.html           단독 실행 공개 시연 화면
app/src/                 편집용 HTML·CSS·JavaScript
data_catalog.csv        데이터 목록·출처·접근 범위
data_lineage.csv        입력 변수·연계키·산출물
```

## 실행 방법

Python 3.12 이상에서 설치합니다.

```bash
pip install .
```

`app/index.html`을 브라우저에서 열면 합성 데이터로 화면과 기능을 살펴볼 수 있습니다. 공개 시연 수치는 실제 지역의 상태를 나타내는 값과 구분합니다. 화면은 `app/src/app.js`, 디자인은 `app/src/styles.css`에서 편집합니다.

승인 집계 3종을 `data/origin/`에 배치한 환경에서는 다음 순서로 분석합니다. 필요한 파일은 해당 폴더의 안내와 `data_catalog.csv`에서 확인할 수 있습니다.

```bash
python src/reproduce.py --source . --output outputs/reproduction
python src/audit_priority.py --source . --output outputs/priority_audit
python src/review_policy.py --input outputs/priority_audit/spatial_quality.csv --output outputs/review_policy
```

센터 전처리 코드는 센터 원자료 접근 환경에서 사용합니다. 노트북 첫 셀의 경로를 설정하고 원 데이터 명세의 소득 구간·순서를 확인합니다.

```bash
pip install jupyterlab
jupyter lab notebooks/center_preprocessing.ipynb
```

공개 시연 데이터를 생성하는 명령입니다.

```bash
python src/build_demo.py --root .
```

## 데이터

승인 집계·지역별 파생값·관측별 예측값은 **보안상 필요 시 승인 범위 내에서 제공합니다.** 경로 안내와 분석 코드는 공개하고, 실제 수치를 담은 자료와 앱은 별도 증빙 패키지에서 관리합니다. 공공자료의 출처는 `data_catalog.csv`와 `results/public_data_sources.json`에 있습니다.

## 결과

지역 회귀는 법정동 116개를 구·군으로 분리해 평가했습니다. MAE는 수도·규모·지역 입력 5.96%p, 소득 포함 입력 6.09%p입니다. 소득은 지역의 경제적 돌봄 여건을 설명하는 자료로 병행합니다.

사용량 예측은 동일 홀짝 검침군의 다음 고지 대표값을 대상으로 합니다. 2023년 7월~2024년 6월 과거 구간의 MAPE는 2.06%입니다. 지역 검토 정책 비교는 정해진 지역 수를 고르는 방식의 차이를 설명하며, 실제 가구 발견 성능이나 예방 효과는 현장 확인 결과로 평가합니다.

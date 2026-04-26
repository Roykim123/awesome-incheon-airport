#!/usr/bin/env python3
"""
mass_backtest_seogeo.py

SEO/GEO 통합 백테스트 스크립트
- rank_correlation: Spearman r 기반 순위 상관관계 분석
- citation_detector: GPT-4o-mini로 AI 응답에서 브랜드 인용 감지
- backlink_analyzer: DA 점수 추정 (규칙 기반 + 테이블 조회)

사용법:
  python mass_backtest_seogeo.py \\
    --brand 걱정마주차 \\
    --keywords 인천공항주차,공항주차대행 \\
    --model gpt-4o-mini \\
    --output results/seogeo_20260426.json

OSS 참조:
  - dipakkr/ai-seo-platform (MIT) — GEO brand visibility platform, gpt-4o-mini provider 구조 참조
  - adulsaa-q/ai_brand_tracker (MIT) — citation rank/position 로직 참조
  - shayanshahravi/serpscore-analyzer — competition score 산식 참조
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai 패키지 필요: pip install openai")

try:
    from scipy.stats import spearmanr
except ImportError:
    raise SystemExit("scipy 필요: pip install scipy")


# ── 설정 ─────────────────────────────────────────────────────────────────────

DEFAULT_BRAND = "걱정마주차"
DEFAULT_KEYWORDS = [
    "인천공항주차",
    "공항주차대행",
    "인천공항주차대행",
    "인천공항장기주차",
    "인천공항 주차장 예약",
]
DEFAULT_MODEL = "gpt-4o-mini"

# GEO 경쟁사 목록 (네이버플레이스 상위 기준)
COMPETITORS = [
    "카카오T주차",
    "AJ파크",
    "공영주차장",
    "파킹클라우드",
    "모두의주차장",
]

# Spearman 비교용 '이상적 순위' (실측값 또는 이전 실행 결과로 갱신)
# keyword -> expected_rank (1=1위, None=미측정)
EXPECTED_RANKS: dict[str, Optional[int]] = {
    "인천공항주차": None,
    "공항주차대행": None,
    "인천공항주차대행": None,
    "인천공항장기주차": None,
    "인천공항 주차장 예약": None,
}


# ── citation_detector ─────────────────────────────────────────────────────────

def citation_detector(
    client: OpenAI,
    keyword: str,
    brand: str,
    competitors: list[str],
    model: str,
) -> dict:
    """
    GPT에게 '키워드로 검색했을 때 어떤 서비스를 추천하는가'를 질의,
    브랜드 언급 여부·순위·포지션을 감지한다.

    OSS 참조: adulsaa-q/ai_brand_tracker collector.py (MIT)

    Returns:
        {
          "keyword": str,
          "mentioned": bool,
          "position": int | None,   # 응답 텍스트 내 문자 위치
          "rank": int | None,       # 언급 순서 (1=first)
          "competitors_mentioned": list[str],
          "raw_response": str,
          "latency_ms": int,
        }
    """
    prompt = (
        f"사용자가 '{keyword}'을(를) 검색했을 때 "
        f"추천할 만한 주차 서비스나 플랫폼을 알려주세요. "
        f"한국 인천공항 근처 기준으로 실제로 존재하는 서비스 이름을 "
        f"순위 형태(1위, 2위, 3위...)로 나열해 주세요."
    )

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "keyword": keyword,
            "mentioned": False,
            "position": None,
            "rank": None,
            "competitors_mentioned": [],
            "raw_response": f"ERROR: {e}",
            "latency_ms": latency_ms,
        }
    latency_ms = int((time.monotonic() - t0) * 1000)

    raw_lower = raw.lower()
    brand_lower = brand.lower()

    # 브랜드 언급 위치
    pos = raw_lower.find(brand_lower)
    mentioned = pos != -1

    # 언급된 모든 서비스 중 순위 계산 (위치 기준 정렬)
    all_services = [brand] + competitors
    service_positions: list[tuple[str, int]] = []
    for svc in all_services:
        p = raw_lower.find(svc.lower())
        if p != -1:
            service_positions.append((svc, p))
    service_positions.sort(key=lambda x: x[1])

    rank: Optional[int] = None
    if mentioned:
        for i, (svc, _) in enumerate(service_positions):
            if svc.lower() == brand_lower:
                rank = i + 1
                break

    competitors_mentioned = [
        svc for svc, _ in service_positions if svc.lower() != brand_lower
    ]

    return {
        "keyword": keyword,
        "mentioned": mentioned,
        "position": pos if mentioned else None,
        "rank": rank,
        "competitors_mentioned": competitors_mentioned,
        "raw_response": raw,
        "latency_ms": latency_ms,
    }


# ── backlink_analyzer ─────────────────────────────────────────────────────────

# 규칙 기반 DA 추정 테이블 (Ahrefs/Moz 없이 근사값)
_DA_TABLE: dict[str, int] = {
    "naver.com": 92,
    "kakao.com": 88,
    "tistory.com": 75,
    "blog.naver.com": 72,
    "dcinside.com": 68,
    "clien.net": 65,
    "ppomppu.co.kr": 62,
    "instiz.net": 58,
    "fmkorea.com": 60,
    "mlbpark.donga.com": 55,
    "theqoo.net": 57,
    "ruliweb.com": 60,
}


def backlink_analyzer(domains: list[str]) -> list[dict]:
    """
    도메인 리스트에 대해 DA 점수를 추정한다.
    알려진 도메인 → 룩업 테이블, 미지 도메인 → 기본값 30.

    OSS 참조: shayanshahravi/serpscore-analyzer HIGH_AUTHORITY_DOMAINS 패턴

    Returns:
        [{"domain": str, "da_score": int, "source": "table"|"table_partial"|"default"}]
    """
    results = []
    for domain in domains:
        domain_clean = domain.lower().strip().rstrip("/")
        # 정확 매치
        if domain_clean in _DA_TABLE:
            results.append({
                "domain": domain_clean,
                "da_score": _DA_TABLE[domain_clean],
                "source": "table",
            })
            continue
        # 부분 매치 (서브도메인 포함)
        matched = False
        for key, score in _DA_TABLE.items():
            if key in domain_clean or domain_clean in key:
                results.append({
                    "domain": domain_clean,
                    "da_score": score,
                    "source": "table_partial",
                })
                matched = True
                break
        if not matched:
            results.append({
                "domain": domain_clean,
                "da_score": 30,
                "source": "default",
            })
    return results


# ── rank_correlation ──────────────────────────────────────────────────────────

def rank_correlation(
    observed: list[Optional[int]],
    expected: list[Optional[int]],
) -> Optional[float]:
    """
    Spearman 순위 상관계수를 계산한다.
    None 값이 있는 키워드는 제외하고 계산.

    Returns:
        float | None  (None=계산 불가 — 유효 데이터 2개 미만)
    """
    pairs = [
        (o, e)
        for o, e in zip(observed, expected)
        if o is not None and e is not None
    ]
    if len(pairs) < 2:
        return None
    obs_vals = [p[0] for p in pairs]
    exp_vals = [p[1] for p in pairs]
    r, _ = spearmanr(obs_vals, exp_vals)
    return round(float(r), 4)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SEO/GEO Mass Backtest (gpt-4o-mini)")
    parser.add_argument("--brand", default=DEFAULT_BRAND, help="추적할 브랜드명")
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="키워드 목록 (쉼표 구분)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI 모델 (기본: gpt-4o-mini)")
    parser.add_argument("--output", default="results/seogeo_latest.json", help="결과 저장 경로")
    parser.add_argument(
        "--backlink-domains",
        default="blog.naver.com,tistory.com,naver.com,kakao.com",
        help="DA 추정할 도메인 목록 (쉼표 구분)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY 환경변수가 없습니다.")

    client = OpenAI(api_key=api_key)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    brand = args.brand

    print(f"[SEO/GEO Backtest] brand={brand}, model={args.model}")
    print(f"  keywords({len(keywords)}): {keywords}")

    # ── 1. citation_detector 실행 ──────────────────────────────────────────
    citation_results: list[dict] = []
    observed_ranks: list[Optional[int]] = []

    for kw in keywords:
        print(f"  [citation] '{kw}' 조회 중...", flush=True)
        res = citation_detector(client, kw, brand, COMPETITORS, args.model)
        status = "O" if res["mentioned"] else "X"
        print(f"    -> mentioned={res['mentioned']} rank={res['rank']} latency={res['latency_ms']}ms [{status}]")
        citation_results.append(res)
        observed_ranks.append(res["rank"])
        time.sleep(1)  # rate limit 방지

    # ── 2. rank_correlation (Spearman r) ───────────────────────────────────
    expected_ranks = [EXPECTED_RANKS.get(kw) for kw in keywords]
    spearman_r = rank_correlation(observed_ranks, expected_ranks)

    if spearman_r is not None:
        print(f"  [rank_correlation] Spearman r = {spearman_r}")
    else:
        print("  [rank_correlation] 기준 순위 데이터 부족 — Spearman r 계산 불가 (첫 실행 시 정상)")

    for row in citation_results:
        row["spearman_r"] = spearman_r

    # ── 3. backlink_analyzer ───────────────────────────────────────────────
    domains = [d.strip() for d in args.backlink_domains.split(",") if d.strip()]
    da_results = backlink_analyzer(domains)
    print(f"  [backlink_analyzer] {da_results}")

    # ── 4. 결과 저장 ───────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_payload = {
        "brand": brand,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": args.model,
        "keywords": keywords,
        "spearman_r_overall": spearman_r,
        "results": citation_results,
        "backlink_da": da_results,
        "oss_references": [
            "dipakkr/ai-seo-platform (MIT) — https://github.com/dipakkr/ai-seo-platform",
            "adulsaa-q/ai_brand_tracker (MIT) — https://github.com/adulsaa-q/ai_brand_tracker",
            "shayanshahravi/serpscore-analyzer — https://github.com/shayanshahravi/serpscore-analyzer",
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    # ── 5. 최종 요약 ───────────────────────────────────────────────────────
    cited_count = sum(1 for r in citation_results if r["mentioned"])
    print(f"\n[완료] 결과 저장: {args.output}")
    print(f"  brand cited: {cited_count}/{len(citation_results)} keywords")
    if spearman_r is not None:
        print(f"  Spearman r = {spearman_r}  (1.0=완벽 상관, -1.0=역상관)")
    print("  DA scores:")
    for da in da_results:
        print(f"    {da['domain']}: {da['da_score']} ({da['source']})")


if __name__ == "__main__":
    main()

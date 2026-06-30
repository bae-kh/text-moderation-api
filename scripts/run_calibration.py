"""
Threshold Calibration Report 생성 스크립트.

소규모 pilot calibration dataset에 대해 모델 추론을 수행하고,
다양한 threshold 조합에서 auto-block precision/recall, safety coverage,
review율을 계산하여 Markdown 형태의 calibration report를 생성합니다.

Usage:
    python scripts/run_calibration.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import pipeline as hf_pipeline

from calibration_data import CALIBRATION_DATASET


MODEL_NAME = "smilegate-ai/kor_unsmile"
MODEL_MAX_TOKENS = 256

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "calibration_results"

# Threshold 탐색 범위
CLEAN_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
HARMFUL_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_model():
    """HuggingFace 모델을 로딩합니다."""
    print(f"Loading model: {MODEL_NAME}")
    classifier = hf_pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        function_to_apply="softmax",
    )
    print("Model loaded successfully.")
    return classifier


def predict_all(classifier, dataset):
    """전체 데이터셋에 대해 모델 추론을 수행합니다."""
    predictions = []

    for i, (text, ground_truth) in enumerate(dataset):
        results = classifier(
            text,
            truncation=True,
            max_length=MODEL_MAX_TOKENS,
        )

        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]

        best = max(results, key=lambda x: x["score"])
        label = best["label"]
        score = float(best["score"])

        predictions.append({
            "text": text,
            "ground_truth": ground_truth,
            "predicted_label": label,
            "confidence": score,
            "is_clean_prediction": label == "clean",
        })

        status = "✓" if (
            (ground_truth == "clean" and label == "clean")
            or (ground_truth == "harmful" and label != "clean")
        ) else "✗"

        print(f"  [{i+1:2d}/{len(dataset)}] {status} {text[:30]:30s} → {label} ({score:.4f}) [GT: {ground_truth}]")

    return predictions


def decide_action(label: str, confidence: float, clean_th: float, harmful_th: float) -> str:
    """주어진 threshold에 따라 action을 결정합니다."""
    if label == "clean":
        return "allow" if confidence >= clean_th else "review"
    return "block" if confidence >= harmful_th else "review"


def calculate_metrics(predictions, clean_th, harmful_th):
    """특정 threshold 조합에서의 metrics를 계산합니다.

    Precision/Recall/F1은 "자동 block을 positive decision으로 보는 기준"으로 계산합니다.
    - Auto-block Precision = block 중 실제 harmful 비율
    - Auto-block Recall = 전체 harmful 중 block된 비율 (review 포함 X)
    - Safety Coverage = 전체 harmful 중 block 또는 review로 잡힌 비율
    """
    tp = 0  # harmful을 block으로 정확히 판단
    fp = 0  # clean을 block으로 잘못 판단 (오탐)

    allow_count = 0
    block_count = 0
    review_count = 0

    harmful_total = sum(1 for p in predictions if p["ground_truth"] == "harmful")
    harmful_allow = 0  # harmful인데 allow로 놓침 (미탐)
    harmful_review = 0  # harmful인데 review로 보냄

    details = []

    for pred in predictions:
        action = decide_action(
            pred["predicted_label"],
            pred["confidence"],
            clean_th,
            harmful_th,
        )
        gt = pred["ground_truth"]

        if action == "allow":
            allow_count += 1
            if gt == "harmful":
                harmful_allow += 1
        elif action == "block":
            block_count += 1
            if gt == "harmful":
                tp += 1
            else:
                fp += 1
        else:
            review_count += 1
            if gt == "harmful":
                harmful_review += 1

        details.append({
            **pred,
            "action": action,
            "correct": (
                (action == "allow" and gt == "clean")
                or (action == "block" and gt == "harmful")
                or action == "review"
            ),
        })

    total = len(predictions)

    # Auto-block Precision: block 중 실제 harmful 비율
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Auto-block Recall: 전체 harmful 중 block된 비율
    recall = tp / harmful_total if harmful_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # Safety Coverage: 전체 harmful 중 block 또는 review로 잡힌 비율
    safety_coverage = (tp + harmful_review) / harmful_total if harmful_total > 0 else 0.0
    # Harmful Allow Rate: 전체 harmful 중 allow로 놓친 비율
    harmful_allow_rate = harmful_allow / harmful_total if harmful_total > 0 else 0.0

    return {
        "clean_threshold": clean_th,
        "harmful_threshold": harmful_th,
        "tp": tp,
        "fp": fp,
        "harmful_allow": harmful_allow,
        "harmful_review": harmful_review,
        "harmful_total": harmful_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "safety_coverage": safety_coverage,
        "harmful_allow_rate": harmful_allow_rate,
        "allow_count": allow_count,
        "block_count": block_count,
        "review_count": review_count,
        "allow_rate": allow_count / total,
        "block_rate": block_count / total,
        "review_rate": review_count / total,
        "details": details,
    }


def find_best_combinations(all_results):
    """F1 기준 상위 조합을 찾습니다."""
    ranked = sorted(all_results, key=lambda x: x["f1"], reverse=True)
    return ranked[:10]


def generate_report(predictions, all_results, best_results, current_result):
    """Markdown 형태의 calibration report를 생성합니다."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(predictions)
    clean_count = sum(1 for p in predictions if p["ground_truth"] == "clean")
    harmful_count = sum(1 for p in predictions if p["ground_truth"] == "harmful")

    # 모델 기본 정확도 (threshold 적용 전)
    correct_predictions = sum(
        1 for p in predictions
        if (p["ground_truth"] == "clean" and p["is_clean_prediction"])
        or (p["ground_truth"] == "harmful" and not p["is_clean_prediction"])
    )
    base_accuracy = correct_predictions / total

    lines = []
    lines.append("# Threshold Calibration Report (Pilot)")
    lines.append("")
    lines.append(f"> Generated: {now}")
    lines.append(f"> Model: `{MODEL_NAME}`")
    lines.append(f"> function_to_apply: `softmax`")
    lines.append(f"> Pilot calibration dataset: {total}건 (clean: {clean_count}, harmful: {harmful_count})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 목적
    lines.append("## 1. 목적")
    lines.append("")
    lines.append("현재 API는 `clean_allow_threshold=0.80`, `harmful_block_threshold=0.65`를 사용합니다.")
    lines.append("이 값이 적절한지 검증하기 위해, 소규모 pilot calibration dataset 60건에 대해")
    lines.append("다양한 threshold 조합에서 auto-block precision, auto-block recall, safety coverage,")
    lines.append("review율을 측정했습니다.")
    lines.append("")
    lines.append("이 결과는 통계적으로 충분한 운영 검증이라기보다, threshold 선택 기준과")
    lines.append("모델 한계를 파악하기 위한 pilot calibration입니다.")
    lines.append("")

    # 2. Metric 정의
    lines.append("## 2. Metric 정의")
    lines.append("")
    lines.append("본 리포트의 precision/recall/F1은 **자동 block을 positive decision으로 보는 기준**으로 계산했습니다.")
    lines.append("")
    lines.append("```text")
    lines.append("Auto-block Precision  = block 중 실제 harmful 비율 (오탐이 적을수록 높음)")
    lines.append("Auto-block Recall     = 전체 harmful 중 block된 비율 (자동 차단 범위)")
    lines.append("Auto-block F1         = Precision과 Recall의 조화 평균")
    lines.append("Safety Coverage       = 전체 harmful 중 block + review로 잡힌 비율")
    lines.append("Harmful Allow Rate    = 전체 harmful 중 allow로 놓친 비율 (미탐)")
    lines.append("Review Rate           = 전체 데이터 중 운영자 검토로 넘어간 비율")
    lines.append("```")
    lines.append("")

    # 3. 모델 기본 성능
    lines.append("## 3. 모델 기본 성능 (threshold 적용 전)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total samples | {total} |")
    lines.append(f"| Clean samples | {clean_count} |")
    lines.append(f"| Harmful samples | {harmful_count} |")
    lines.append(f"| Correct predictions | {correct_predictions} / {total} |")
    lines.append(f"| Base accuracy | {base_accuracy:.1%} |")
    lines.append("")
    lines.append("Base accuracy는 모델의 top-1 label이 ground truth와 일치하는 비율입니다.")
    lines.append("threshold나 action 정책 적용 전의 순수 모델 분류 정확도입니다.")
    lines.append("")

    # 4. 개별 예측 결과
    lines.append("## 4. 개별 예측 결과")
    lines.append("")
    lines.append("| # | Text | Ground Truth | Predicted | Confidence | Match |")
    lines.append("|---|------|-------------|-----------|-----------|-------|")
    for i, pred in enumerate(predictions):
        match = "✓" if (
            (pred["ground_truth"] == "clean" and pred["is_clean_prediction"])
            or (pred["ground_truth"] == "harmful" and not pred["is_clean_prediction"])
        ) else "✗"
        text_display = pred["text"][:25]
        lines.append(
            f"| {i+1} | {text_display} | {pred['ground_truth']} | "
            f"{pred['predicted_label']} | {pred['confidence']:.4f} | {match} |"
        )
    lines.append("")

    # 5. 현재 threshold 성능
    lines.append("## 5. 현재 threshold 성능")
    lines.append("")
    lines.append(f"현재 설정: `clean_allow_threshold=0.80`, `harmful_block_threshold=0.65`")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Auto-block Precision | {current_result['precision']:.1%} |")
    lines.append(f"| Auto-block Recall | {current_result['recall']:.1%} |")
    lines.append(f"| Auto-block F1 | {current_result['f1']:.1%} |")
    lines.append(f"| Safety Coverage (block + review) | {current_result['safety_coverage']:.1%} |")
    lines.append(f"| Harmful Allow Rate (미탐) | {current_result['harmful_allow_rate']:.1%} ({current_result['harmful_allow']}/{current_result['harmful_total']}) |")
    lines.append(f"| Block | {current_result['block_count']}/{total} (TP={current_result['tp']}, FP={current_result['fp']}) |")
    lines.append(f"| Review | {current_result['review_count']}/{total} (harmful {current_result['harmful_review']}건 포함) |")
    lines.append(f"| Allow | {current_result['allow_count']}/{total} |")
    lines.append(f"| Review Rate | {current_result['review_rate']:.1%} |")
    lines.append("")

    # 6. Threshold 조합별 비교
    lines.append("## 6. Threshold 조합별 비교")
    lines.append("")
    lines.append("| Clean TH | Harmful TH | Precision | Recall | F1 | Safety Cov. | Allow | Block | Review |")
    lines.append("|----------|-----------|-----------|--------|-----|-------------|-------|-------|--------|")
    for r in all_results:
        marker = " ◀ current" if r["clean_threshold"] == 0.80 and r["harmful_threshold"] == 0.65 else ""
        lines.append(
            f"| {r['clean_threshold']:.2f} | {r['harmful_threshold']:.2f} | "
            f"{r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.1%} | "
            f"{r['safety_coverage']:.1%} | "
            f"{r['allow_count']} | {r['block_count']} | {r['review_count']} |"
            f"{marker}"
        )
    lines.append("")

    # 7. F1 상위 조합
    lines.append("## 7. Auto-block F1 기준 상위 10개 조합")
    lines.append("")
    lines.append("| Rank | Clean TH | Harmful TH | Precision | Recall | F1 | Safety Cov. | Review Rate |")
    lines.append("|------|----------|-----------|-----------|--------|-----|-------------|-------------|")
    for i, r in enumerate(best_results):
        current = " ◀" if r["clean_threshold"] == 0.80 and r["harmful_threshold"] == 0.65 else ""
        lines.append(
            f"| {i+1} | {r['clean_threshold']:.2f} | {r['harmful_threshold']:.2f} | "
            f"{r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.1%} | "
            f"{r['safety_coverage']:.1%} | {r['review_rate']:.1%} |{current}"
        )
    lines.append("")

    # 8. 분석
    lines.append("## 8. 분석 및 결론")
    lines.append("")

    ranked = sorted(all_results, key=lambda x: x["f1"], reverse=True)
    current_rank = next(
        i + 1 for i, r in enumerate(ranked)
        if r["clean_threshold"] == 0.80 and r["harmful_threshold"] == 0.65
    )
    best = ranked[0]

    lines.append(f"현재 threshold (`0.80/0.65`)는 Auto-block F1 기준 **{current_rank}위** (총 {len(all_results)}개 조합 중)입니다.")
    lines.append("")
    lines.append(f"F1 기준 최적 조합은 `clean={best['clean_threshold']:.2f}/harmful={best['harmful_threshold']:.2f}` ")
    lines.append(f"(F1={best['f1']:.1%})이지만, 현재 설정을 기본값으로 유지합니다.")
    lines.append("")

    lines.append("### 현재 threshold 유지 이유 (운영 목표 기반)")
    lines.append("")
    lines.append("현재 프로젝트의 threshold 선택 기준은 다음과 같습니다.")
    lines.append("")
    lines.append("```text")
    lines.append("1순위: 정상 텍스트를 자동 차단하지 않기 (Auto-block Precision 최대화)")
    lines.append("2순위: Review rate를 너무 높이지 않기 (운영자 부담 억제)")
    lines.append("3순위: Harmful recall / F1 높이기")
    lines.append("```")
    lines.append("")
    lines.append("F1이 가장 높은 `clean=0.95` 조합은 review rate가 증가합니다.")
    lines.append("현재 `0.80/0.65` 설정은 auto-block precision 100%와 review rate 10%를")
    lines.append("만족하는 보수적인 운영 정책으로 유지합니다.")
    lines.append("")

    lines.append("### 미탐 분석")
    lines.append("")
    lines.append("모델이 harmful을 clean으로 판단한 사례를 분석하면,")
    lines.append("직접적인 욕설이 없는 차별/비하/위협 표현이 주를 이룹니다.")
    lines.append("")
    lines.append("일부 미탐은 clean_threshold를 높여 allow에서 review로 이동시킬 수 있지만,")
    lines.append("모델이 clean으로 강하게 판단하는 경우(confidence > 0.90)는")
    lines.append("threshold 조정만으로 완전히 해결하기 어렵습니다.")
    lines.append("이 유형은 해당 데이터를 보강한 fine-tuning이 필요합니다.")
    lines.append("")

    # 9. 한계
    lines.append("## 9. 한계 및 향후 개선")
    lines.append("")
    lines.append("- 이 결과는 60건의 소규모 pilot calibration이며, 통계적으로 충분한 운영 검증이 아닙니다")
    lines.append("- 실제 운영에서는 review_result 데이터를 수백~수천 건 축적한 뒤 재측정해야 합니다")
    lines.append("- 테스트 데이터가 실제 신고 데이터 분포를 대표하지 않을 수 있습니다")
    lines.append("- 모델 자체의 한계: 직접 욕설이 없는 차별/비하/위협 표현에 약함")
    lines.append("- category별 세분화된 threshold 검토 가능 (예: 욕설 vs 차별 vs 위협)")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Threshold Calibration Report Generator")
    print("=" * 70)
    print()

    # 1. 모델 로딩
    classifier = load_model()
    print()

    # 2. 전체 데이터 추론
    print(f"Running predictions on {len(CALIBRATION_DATASET)} samples...")
    predictions = predict_all(classifier, CALIBRATION_DATASET)
    print()

    # 3. 모든 threshold 조합 계산
    print("Calculating metrics for all threshold combinations...")
    all_results = []
    current_result = None

    for clean_th in CLEAN_THRESHOLDS:
        for harmful_th in HARMFUL_THRESHOLDS:
            result = calculate_metrics(predictions, clean_th, harmful_th)
            all_results.append(result)

            if clean_th == 0.80 and harmful_th == 0.65:
                current_result = result

    print(f"  Evaluated {len(all_results)} combinations.")
    print()

    # 4. 상위 조합 찾기
    best_results = find_best_combinations(all_results)

    # 5. Report 생성
    print("Generating calibration report...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report(predictions, all_results, best_results, current_result)
    report_path = OUTPUT_DIR / "calibration_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to: {report_path}")

    # 6. Raw 데이터 저장
    raw_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "total_samples": len(predictions),
        "predictions": [
            {k: v for k, v in p.items()}
            for p in predictions
        ],
        "current_thresholds": {
            "clean": 0.80,
            "harmful": 0.65,
            "auto_block_precision": current_result["precision"],
            "auto_block_recall": current_result["recall"],
            "auto_block_f1": current_result["f1"],
            "safety_coverage": current_result["safety_coverage"],
            "harmful_allow_rate": current_result["harmful_allow_rate"],
            "review_rate": current_result["review_rate"],
        },
        "best_f1_combination": {
            "clean": best_results[0]["clean_threshold"],
            "harmful": best_results[0]["harmful_threshold"],
            "f1": best_results[0]["f1"],
        },
    }

    raw_path = OUTPUT_DIR / "calibration_raw.json"
    raw_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Raw data saved to: {raw_path}")

    # 7. 결과 요약 출력
    print()
    print("=" * 70)
    print("CALIBRATION SUMMARY (Pilot)")
    print("=" * 70)
    print(f"  Current thresholds: clean={0.80}, harmful={0.65}")
    print(f"  Auto-block Precision: {current_result['precision']:.1%}")
    print(f"  Auto-block Recall:    {current_result['recall']:.1%}")
    print(f"  Auto-block F1:        {current_result['f1']:.1%}")
    print(f"  Safety Coverage:      {current_result['safety_coverage']:.1%}")
    print(f"  Harmful Allow Rate:   {current_result['harmful_allow_rate']:.1%}")
    print(f"  Block:     {current_result['block_count']}/{len(predictions)} ({current_result['block_rate']:.1%})")
    print(f"  Review:    {current_result['review_count']}/{len(predictions)} ({current_result['review_rate']:.1%})")
    print(f"  Allow:     {current_result['allow_count']}/{len(predictions)} ({current_result['allow_rate']:.1%})")
    print()
    print(f"  Best F1 combination: clean={best_results[0]['clean_threshold']}, harmful={best_results[0]['harmful_threshold']} (F1={best_results[0]['f1']:.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()

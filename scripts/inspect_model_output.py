from typing import Any

from transformers import pipeline

MODEL_NAME = "smilegate-ai/kor_unsmile"


def normalize_results(results: Any) -> list[dict[str, Any]]:
    """
    transformers pipeline 결과 구조가 버전에 따라
    flat list 또는 nested list로 나올 수 있어 이를 정규화한다.
    """
    if isinstance(results, list) and len(results) > 0:
        if isinstance(results[0], dict):
            return results

        if isinstance(results[0], list):
            return results[0]

    raise ValueError(f"Unexpected pipeline output format: {results}")


def main() -> None:
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        function_to_apply="sigmoid",
    )

    texts = [
        "테스트 문장입니다.",
        "바보",
        "바보같아",
        "아쉽게 생겼네",
        "바보같은놈",
        "병신같은놈 나가죽어 왤케 못생겼어 이거 완전 미친놈이잖아",
        "야 이 씹새끼야 뭐하냐 나가 죽어라"
    ]

    for text in texts:
        print("=" * 80)
        print("TEXT:", text)

        raw_results = classifier(
            text,
            truncation=True,
            max_length=256,
        )

        results = normalize_results(raw_results)

        sorted_results = sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )

        for item in sorted_results:
            print(f'{item["label"]}: {item["score"]:.4f}')


if __name__ == "__main__":
    main()

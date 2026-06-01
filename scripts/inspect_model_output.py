from transformers import pipeline


MODEL_NAME = "smilegate-ai/kor_unsmile"
MODEL_MAX_TOKENS = 256


def main() -> None:
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        function_to_apply="softmax",
    )

    texts = [
        "테스트 문장입니다.",
        "바보",
        "바보같아",
        "아쉽게 생겼네",
        "바보같은놈",
    ]

    for text in texts:
        print("=" * 80)
        print("TEXT:", text)

        results = classifier(
            text,
            truncation=True,
            max_length=MODEL_MAX_TOKENS,
        )

        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]

        sorted_results = sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )

        for item in sorted_results:
            print(f'{item["label"]}: {item["score"]:.4f}')


if __name__ == "__main__":
    main()

import csv
from pathlib import Path


RESULT_DIR = Path("load_test_results")
SCENARIOS = [1, 5, 10, 20]


def find_aggregated_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        rows = list(reader)

    # Locust CSV에는 보통 Aggregated row가 있음
    for row in rows:
        if row.get("Name") == "Aggregated":
            return row

    # Aggregated가 없으면 마지막 row를 fallback으로 사용
    if rows:
        return rows[-1]

    raise ValueError(f"No rows found in {csv_path}")


def to_float(value: str) -> float:
    if value is None or value == "":
        return 0.0

    return float(value)


def main() -> None:
    summary_rows: list[dict[str, str]] = []

    for users in SCENARIOS:
        csv_path = RESULT_DIR / f"users_{users}_stats.csv"

        if not csv_path.exists():
            print(f"Missing file: {csv_path}")
            continue

        row = find_aggregated_row(csv_path)

        request_count = to_float(row.get("Request Count", "0"))
        failure_count = to_float(row.get("Failure Count", "0"))

        failure_rate = 0.0
        if request_count > 0:
            failure_rate = (failure_count / request_count) * 100

        summary_rows.append({
            "Users": str(users),
            "Avg Latency (ms)": row.get("Average Response Time", "0"),
            "p95 Latency (ms)": row.get("95%", "0"),
            "p99 Latency (ms)": row.get("99%", "0"),
            "RPS": row.get("Requests/s", "0"),
            "Failure Rate (%)": f"{failure_rate:.2f}",
        })

    output_path = RESULT_DIR / "summary.md"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Load Test Summary\n\n")
        file.write("## Results\n\n")
        file.write("| Users | Avg Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | RPS | Failure Rate (%) |\n")
        file.write("|---:|---:|---:|---:|---:|---:|\n")

        for row in summary_rows:
            file.write(
                f'| {row["Users"]} '
                f'| {row["Avg Latency (ms)"]} '
                f'| {row["p95 Latency (ms)"]} '
                f'| {row["p99 Latency (ms)"]} '
                f'| {row["RPS"]} '
                f'| {row["Failure Rate (%)"]} |\n'
            )

    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
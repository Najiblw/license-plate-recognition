from __future__ import annotations

import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


FILES = [
    ("Armenian-license-plate.jpg", "armenian_plate.jpg"),
    ("Monaco vehicle registration plate.jpg", "monaco_plate.jpg"),
    ("German electric car license plate.jpg", "german_electric_plate.jpg"),
    ("Private vehicle plate palestine.JPG", "palestine_plate.jpg"),
    ("J 1 - Jersey 1 front plate.jpg", "jersey_front_plate.jpg"),
    ("2007 Queensland registration plate 01 SWIM vanity.jpg", "queensland_swim_plate.jpg"),
    ("2007 Queensland registration plate 01 SWIM personal on Lexus.jpg", "queensland_lexus_car.jpg"),
]


def build_url(filename: str) -> str:
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}"


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def main() -> None:
    target_dir = Path(__file__).resolve().parents[1] / "data" / "input"
    target_dir.mkdir(parents=True, exist_ok=True)

    for source_name, output_name in FILES:
        destination = target_dir / output_name
        if destination.exists():
            print(f"SKIP {output_name}")
            continue

        url = build_url(source_name)
        success = False

        for attempt in range(3):
            try:
                download_file(url, destination)
                print(f"DOWNLOADED {output_name}")
                success = True
                break
            except HTTPError as error:
                print(f"HTTP {error.code} {output_name} attempt={attempt + 1}")
            except URLError as error:
                print(f"URLERROR {output_name} attempt={attempt + 1} error={error.reason}")

            time.sleep(3 * (attempt + 1))

        if not success:
            print(f"FAILED {output_name}")


if __name__ == "__main__":
    main()

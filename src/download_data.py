"""Baixa os arquivos do dataset Reddit Hyperlink Network (SNAP)."""

from pathlib import Path
import urllib.request

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = {
    "soc-redditHyperlinks-body.tsv": "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
    "soc-redditHyperlinks-title.tsv": "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] Arquivo já existe: {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)")
        return

    print(f"[BAIXANDO] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_dest, "wb") as out:
        chunk_size = 1024 * 1024
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
    tmp_dest.replace(dest)
    print(f"[CONCLUÍDO] {dest.name} ({downloaded / 1e6:.2f} MB)")


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        download_file(url, DATA_RAW_DIR / filename)


if __name__ == "__main__":
    main()

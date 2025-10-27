import requests
from bs4 import BeautifulSoup

def fetch_website(url: str, timeout: int = 10) -> str:
    res = requests.get(url, timeout=timeout)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text().split())

    return text[:15000]

"""通用计算机知识网页抓取器（用于构建计算机知识图谱语料）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class KnowledgeRecord:
    title: str
    url: str
    summary: str


class ComputerKnowledgeScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            }
        )

    def fetch(self, url: str) -> str | None:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except Exception:
            return None

    def parse_article(self, html: str, url: str) -> KnowledgeRecord:
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.get_text(strip=True) if soup.title else "Untitled")
        text = " ".join(x.get_text(" ", strip=True) for x in soup.find_all(["p", "li"]))
        text = re.sub(r"\s+", " ", text).strip()
        summary = text[:1200]
        return KnowledgeRecord(title=title, url=url, summary=summary)

    def scrape(self, urls: list[str]) -> list[dict]:
        records: list[dict] = []
        for url in urls:
            html = self.fetch(url)
            if not html:
                continue
            rec = self.parse_article(html, url)
            records.append({"title": rec.title, "source_url": rec.url, "description": rec.summary})
        return records

    @staticmethod
    def save_json(records: list[dict], output_file: str = "computer_knowledge_corpus.json") -> str:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return output_file
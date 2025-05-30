import asyncio
import json
import random
import ssl
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import certifi
from aiohttp import TCPConnector
from bs4 import BeautifulSoup


class WebScraper:
    """Unified web scraper for processing search results with different extraction modes."""

    def __init__(
        self,
        extraction_mode: str = "query_window",
        max_articles_per_query: int = None,
        fragments_per_article: int = 10,
        chars_per_fragment: int = 500,
        window_size: int = 500,
        delay_range: Tuple[float, float] = (1, 10),
    ):
        """
        Initialize the web scraper.

        Args:
            extraction_mode: Either "query_window" or "random_fragments"
            max_articles_per_query: Limit number of articles per query (None for no limit)
            fragments_per_article: Number of random fragments to extract per article
            chars_per_fragment: Characters per fragment for random extraction
            window_size: Size of text window around query terms
            delay_range: Min and max delay between requests (in seconds)
        """
        self.extraction_mode = extraction_mode
        self.max_articles_per_query = max_articles_per_query
        self.fragments_per_article = fragments_per_article
        self.chars_per_fragment = chars_per_fragment
        self.window_size = window_size
        self.delay_range = delay_range

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def load_search_results(self, file_path: str) -> List[Dict]:
        """Load JSON file with Google search results."""
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def extract_query(self, search_result: Dict) -> str:
        """Extract the query from search parameters."""
        return search_result.get("searchParameters", {}).get("q", "")

    def extract_links(self, search_result: Dict) -> List[str]:
        """Extract links from organic search results."""
        links = []
        for item in search_result.get("organic", []):
            link = item.get("link")
            if link:
                links.append(link)
            if (
                self.max_articles_per_query
                and len(links) >= self.max_articles_per_query
            ):
                break
        return links

    def extract_text_window(self, text: str, index: int, window_size: int) -> str:
        """Extract a text window of specified size around an index."""
        text_length = len(text)
        half_window = window_size // 2

        # If first occurrence is too close to the beginning
        if index < half_window:
            return text[:window_size] if text_length >= window_size else text

        # If first occurrence is too close to the end
        elif index > text_length - half_window:
            start = text_length - window_size
            start = max(0, start)  # Ensure start isn't negative
            return text[start:] if text_length >= window_size else text

        # Normal case: extract window with occurrence in the middle
        else:
            start = index - half_window
            end = start + window_size
            return text[start:end]

    def split_into_char_fragments(
        self, text: str, num_fragments: int, chars_per_fragment: int
    ) -> List[str]:
        """Split text into random fragments of specified character length."""
        total_chars = len(text)
        fragments = []
        if total_chars < chars_per_fragment:
            return fragments

        for _ in range(num_fragments):
            start = random.randint(0, total_chars - chars_per_fragment)
            frag_text = text[start : start + chars_per_fragment]
            fragments.append(frag_text)
        return fragments

    async def scrape_webpage(
        self, session: aiohttp.ClientSession, url: str, query: str = ""
    ) -> Optional[Dict]:
        """Scrape a single webpage and extract content based on extraction mode."""
        try:
            # Add random delay to avoid overloading servers
            await asyncio.sleep(random.uniform(*self.delay_range))

            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return None

            async with session.get(url, headers=self.headers, timeout=15) as response:
                if self.extraction_mode == "random_fragments":
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        return None

                if response.status != 200:
                    return None

                html = await response.text()

            soup = BeautifulSoup(html, "html.parser")
            body = soup.body
            if not body:
                return None

            text = body.get_text(separator=" ", strip=True)

            if self.extraction_mode == "query_window":
                if not query:
                    return None

                query_words = query.split()
                first_occurrence = None
                first_index = float("inf")

                for word in query_words:
                    index = text.find(word)
                    if index != -1 and index < first_index:
                        first_index = index
                        first_occurrence = word

                if first_occurrence is None:
                    return None

                # Extract window of text around the first occurrence
                text_window = self.extract_text_window(
                    text, first_index, self.window_size
                )
                return {"link": url, "text": text_window}

            elif self.extraction_mode == "random_fragments":
                fragments = self.split_into_char_fragments(
                    text, self.fragments_per_article, self.chars_per_fragment
                )
                if not fragments:
                    return None
                return {"link": url, "fragments": fragments}

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    async def process_query(
        self, session: aiohttp.ClientSession, search_result: Dict
    ) -> Tuple[str, List[Dict]]:
        """Process a single query asynchronously."""
        query = self.extract_query(search_result)
        links = self.extract_links(search_result)

        print(f"Processing query: {query} with {len(links)} links")

        if self.extraction_mode == "query_window":
            tasks = [self.scrape_webpage(session, link, query) for link in links]
        else:
            tasks = [self.scrape_webpage(session, link) for link in links]

        results = await asyncio.gather(*tasks)

        valid_results = [result for result in results if result]

        return query, valid_results

    async def scrape_dataset(self, input_file: str, output_file: str):
        """Main function to handle the entire scraping process."""
        search_results = self.load_search_results(input_file)
        output_data = {}

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector) as session:
            for search_result in search_results:
                query, results = await self.process_query(session, search_result)
                if query and results:
                    output_data[query] = results

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(output_data, file, ensure_ascii=False, indent=2)

        print(f"Results saved to {output_file}")


async def main():
    # For the dataset of positive examples (query window extraction)
    scraper1 = WebScraper(
        extraction_mode="query_window", window_size=500, delay_range=(1, 10)
    )

    # For the dataset of negative examples (random fragments extraction)
    scraper2 = WebScraper(
        extraction_mode="random_fragments",
        max_articles_per_query=10,
        fragments_per_article=10,
        chars_per_fragment=500,
        delay_range=(1, 3),
    )

    await scraper1.scrape_dataset(
        "data/datasets/politicians_search_results.json",
        "data/datasets/extracted_text_results.json",
    )

    await scraper2.scrape_dataset(
        "data/datasets/test_search_results_with_links.json",
        "data/datasets/extracted_test_results.json",
    )


if __name__ == "__main__":
    asyncio.run(main())

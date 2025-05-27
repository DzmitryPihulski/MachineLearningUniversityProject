import asyncio
import json
import random
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup


# Function to load the JSON file with Google search results
def load_search_results(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# Function to extract the query from search parameters
def extract_query(search_result: Dict) -> str:
    return search_result.get("searchParameters", {}).get("q", "")


# Function to extract links from organic search results
def extract_links(search_result: Dict) -> List[str]:
    links = []
    for item in search_result.get("organic", []):
        link = item.get("link")
        if link:
            links.append(link)
    return links


# Function to extract a text window of specified size around an index
def extract_text_window(text: str, index: int, window_size: int) -> str:
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


# Function to scrape a single webpage
async def scrape_webpage(
    session: aiohttp.ClientSession, url: str, query: str
) -> Optional[Dict]:
    try:
        # Add a random delay to avoid overloading servers
        await asyncio.sleep(random.uniform(1, 10))

        # Parse the URL and check if it's valid
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status != 200:
                return None

            html = await response.text()

            # Parse HTML and extract body text
            soup = BeautifulSoup(html, "html.parser")
            body = soup.body
            if not body:
                return None

            # Get text content from body
            text = body.get_text(separator=" ", strip=True)

            # Find query terms in the text (case sensitive)
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
            text_window = extract_text_window(text, first_index, 500)

            return {"link": url, "text": text_window}

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None


# Function to process a single query asynchronously
async def process_query(
    session: aiohttp.ClientSession, search_result: Dict
) -> Tuple[str, List[Dict]]:
    query = extract_query(search_result)
    links = extract_links(search_result)

    print(f"Processing query: {query} with {len(links)} links")

    # Scrape all links for this query
    tasks = [scrape_webpage(session, link, query) for link in links]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    valid_results = [result for result in results if result]

    return query, valid_results


# Main function to handle the entire process
async def main(input_file: str, output_file: str):
    search_results = load_search_results(input_file)
    output_data = {}

    # Create a single session for all requests
    async with aiohttp.ClientSession() as session:
        # Process queries one at a time (all links for a query in parallel)
        for search_result in search_results:
            query, results = await process_query(session, search_result)
            if query and results:
                output_data[query] = results

    # Save the results to a JSON file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")


# Run the script
if __name__ == "__main__":
    INPUT_FILE = "data/datasets/politicians_search_results.json"  # Change this to your input file path
    OUTPUT_FILE = "data/datasets/extracted_test_results.json"  # Output file path

    # Run the async main function
    asyncio.run(main(INPUT_FILE, OUTPUT_FILE))

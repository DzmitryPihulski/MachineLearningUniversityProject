import json
import os
import time
import requests

SERPER_API_KEY = os.getenv("SERPER_API_KEY") or "ENTER_KEY"

INPUT_FILE = "test_search_queries.json"
OUTPUT_FILE = "test_search_results_with_links.json"

SERPER_URL = "https://google.serper.dev/search"

def fetch_links_for_query(query, hl="en", gl="us", num=10):
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "hl": hl,
        "gl": gl,
        "num": num
    }
    response = requests.post(SERPER_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Błąd {response.status_code} dla zapytania: {query}")
        return None

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []

    for entry in queries:
        params = entry.get("searchParameters", {})
        query = params.get("q")
        hl = params.get("hl", "en")
        gl = params.get("gl", "us")
        num = params.get("num", 10)

        print(f"Pobieranie linków dla: {query}")
        search_result = fetch_links_for_query(query, hl, gl, num)
        if search_result:
            entry["organic"] = search_result.get("organic", [])
            entry["knowledgeGraph"] = search_result.get("knowledgeGraph", {})

        results.append(entry)
        time.sleep(1.5) 

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Zapisano {len(results)} wyników do {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
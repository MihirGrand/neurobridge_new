"""News tool - fetch latest headlines via the Currents API."""

import os

import requests


def get_news(category: str = "general") -> str:
    """
    Fetch the latest headlines from around the world and India using the
    free Currents API (currentsapi.services).

    Args:
        category: One of the Currents API categories: "general", "society",
                  "science_technology", "politics_government",
                  "economy_business_finance", "arts_culture_entertainment",
                  "lifestyle_leisure", "human_interest", "sport",
                  "crime_law_justice", "education", "environment",
                  "labour", "health", "automotive", "real_estate".
                  Common aliases like "business", "technology", "sports",
                  "entertainment", "health", "science" also work.

    Returns:
        A formatted string with headlines and short descriptions.
    """
    api_key = os.getenv("CURRENTS_API_KEY")
    if not api_key:
        return "ERROR: CURRENTS_API_KEY environment variable is not set."

    base_url = "https://api.currentsapi.services/v1/latest-news"
    headers = {"Authorization": api_key}
    results: list[str] = []

    # Map common shorthand aliases to Currents category names
    category_aliases = {
        "general": "general",
        "business": "economy_business_finance",
        "technology": "science_technology",
        "sports": "sport",
        "entertainment": "arts_culture_entertainment",
        "health": "health",
        "science": "science_technology",
    }
    currents_category = category_aliases.get(category, category)

    for label, params in [
        (
            "World News",
            {"language": "en", "category": currents_category, "page_size": 5},
        ),
        (
            "India News",
            {
                "language": "en",
                "category": currents_category,
                "country": "IN",
                "page_size": 5,
            },
        ),
    ]:
        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            results.append(f"\n{label}: Failed to fetch - {e}")
            continue

        if data.get("status") != "ok":
            results.append(
                f"\n{label}: API returned error - {data.get('message', 'unknown')}"
            )
            continue

        articles = data.get("news", [])
        if not articles:
            results.append(f"\n{label}: No articles found.")
            continue

        lines = [f"\n{label}:"]
        for i, a in enumerate(articles, 1):
            title = a.get("title", "(no title)")
            desc = a.get("description", "")
            author = a.get("author", "")
            url = a.get("url", "")
            published = a.get("published", "")
            meta = f" ({author})" if author else ""
            line = f"  {i}. {title}{meta}"
            if published:
                line += f" [{published}]"
            if desc:
                line += f"\n     {desc}"
            line += f"\n     Link: {url}"
            lines.append(line)
        results.append("\n".join(lines))

    return "\n".join(results)

# shared_tools/scraper_tool.py

import requests
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
import yaml

from langchain_core.tools import tool

from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

class ScraperTool:
    def __init__(self, config_manager, user_manager):
        self.config_manager = config_manager
        self.user_manager = user_manager

    def _get_search_api_key(self, api_name: str) -> Optional[str]:
        """
        Retrieves the API key for a given search API from secrets.
        """
        if api_name == "serpapi":
            return self.config_manager.get_secret("serpapi_api_key")
        elif api_name == "google_custom_search":
            return self.config_manager.get_secret("google_custom_search_api_key")
        return None

    def _format_search_results(self, results: List[Dict[str, str]], max_chars: int) -> str:
        """
        Formats the list of search results into a readable string, truncating snippets.
        """
        formatted_output = []
        for i, res in enumerate(results):
            snippet = res.get("snippet", "No snippet available.")
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."

            formatted_output.append(
                f"Result {i+1}:\n"
                f"Title: {res.get('title', 'N/A')}\n"
                f"Link: {res.get('link', 'N/A')}\n"
                f"Snippet: {snippet}\n"
                f"---"
            )
        return "\n".join(formatted_output)

    @tool
    def scrape_web(self, query: str, user_context: Optional[UserProfile] = None, max_chars: Optional[int] = None) -> str:
        """
        Searches the web for information using a smart search fallback mechanism.
        It attempts to use configured search APIs (like SerpAPI or Google Custom Search) first.
        If no API key is available or the API call fails, it falls back to direct web scraping
        of a general search engine (e.g., Google Search results page).

        Args:
            query (str): The search query.
            user_context (UserProfile, optional): The user's profile for RBAC capability checks.
                                                  Defaults to None.
            max_chars (int, optional): Maximum characters for the returned snippet.
                                       If not provided, it will be determined by user's tier capability.

        Returns:
            str: A string containing relevant information from the web, or an error message.
        """
        user_id = user_context.user_id if user_context else "default"
        user_tier = user_context.tier if user_context else "default"

        logger.info(f"Tool: scrape_web called with query: '{query}' for user: '{user_id}' (tier: {user_tier})")

        if not query:
            return "Please provide a non-empty query."

        # Get user's allowed max_chars from RBAC capabilities if not explicitly provided
        if max_chars is None:
            max_chars = self.user_manager.get_user_tier_capability(user_id, 'web_search_limit_chars', self.config_manager.get('web_scraping.max_search_results', 500), user_tier=user_tier)

        # Get max search results allowed by user's tier
        max_results_allowed = self.user_manager.get_user_tier_capability(user_id, 'web_search_max_results', self.config_manager.get('web_scraping.max_search_results', 5), user_tier=user_tier)

        headers = {
            "User-Agent": self.config_manager.get("web_scraping.user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "DNT": "1", # Do Not Track
            "Connection": "keep-alive",
        }
        timeout = self.config_manager.get("web_scraping.timeout_seconds", 15)

        search_results = []

        # --- Attempt to use configured Search APIs first (e.g., SerpAPI, Google Custom Search) ---
        search_apis = self.config_manager.get("api_configs", [])
        for api_config_file in search_apis:
            api_path = Path(f"data/{api_config_file}")
            if not api_path.exists():
                continue
            try:
                with open(api_path, "r") as f:
                    full_api_config = yaml.safe_load(f) or {}
                    for api_info in full_api_config.get('search_apis', []):
                        api_name = api_info.get("name")
                        api_type = api_info.get("type")
                        if api_type == "search":
                            api_key = self._get_search_api_key(api_name.lower())
                            if api_key:
                                logger.info(f"Attempting to use {api_name} for web search.")
                                try:
                                    if api_name.lower() == "serpapi":
                                        params = {
                                            "api_key": api_key,
                                            "q": query,
                                            "engine": "google",
                                            "num": min(10, max_results_allowed)
                                        }
                                        response = requests.get("https://serpapi.com/search", params=params, timeout=timeout)
                                        response.raise_for_status()
                                        data = response.json()
                                        if "organic_results" in data:
                                            for res in data["organic_results"][:max_results_allowed]:
                                                search_results.append({
                                                    "title": res.get("title"),
                                                    "link": res.get("link"),
                                                    "snippet": res.get("snippet")
                                                })
                                            if search_results:
                                                logger.info(f"Successfully fetched {len(search_results)} results from SerpAPI.")
                                                return self._format_search_results(search_results, max_chars)

                                    elif api_name.lower() == "google_custom_search":
                                        cx = self.config_manager.get_secret("google_custom_search_cx")
                                        if not cx:
                                            logger.warning("Google Custom Search CX not found in secrets. Skipping.")
                                            continue
                                        params = {
                                            "key": api_key,
                                            "cx": cx,
                                            "q": query,
                                            "num": min(10, max_results_allowed)
                                        }
                                        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=timeout)
                                        response.raise_for_status()
                                        data = response.json()
                                        if "items" in data:
                                            for res in data["items"][:max_results_allowed]:
                                                search_results.append({
                                                    "title": res.get("title"),
                                                    "link": res.get("link"),
                                                    "snippet": res.get("snippet")
                                                })
                                            if search_results:
                                                logger.info(f"Successfully fetched {len(search_results)} results from Google Custom Search.")
                                                return self._format_search_results(search_results, max_chars)

                                except requests.exceptions.RequestException as req_e:
                                    logger.warning(f"API search with {api_name} failed: {req_e}. Falling back to direct scraping.")
                                except Exception as e:
                                    logger.warning(f"Error processing {api_name} response: {e}. Falling back to direct scraping.")
            except Exception as e:
                logger.error(f"Error loading API config from {api_path}: {e}")
                continue

        # --- Fallback to direct Google Search scraping if no API works or is configured ---
        logger.info("Falling back to direct Google Search scraping.")
        try:
            search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            response = requests.get(search_url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            for g in soup.find_all('div', class_='g')[:max_results_allowed]:
                title_tag = g.find('h3')
                link_tag = g.find('a')
                snippet_tag = g.find('div', class_='VwiC3b')

                title = title_tag.get_text() if title_tag else "No Title"
                link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else "No Link"
                snippet = snippet_tag.get_text() if snippet_tag else "No Snippet"

                search_results.append({"title": title, "link": link, "snippet": snippet})

            if search_results:
                logger.info(f"Successfully scraped {len(search_results)} results from Google Search.")
                return self._format_search_results(search_results, max_chars)
            else:
                logger.warning("No search results found via direct scraping.")
                return "No relevant information found on the web."

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to scrape web: {e}", exc_info=True)
            return f"Failed to perform web search due to a network error: {e}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during web scraping: {e}", exc_info=True)
            return f"An unexpected error occurred during web search: {e}"

# backend/services/llm_service.py

import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status, Depends # ADDED Depends here
from datetime import datetime, timedelta, timezone
import asyncio # Import asyncio to check for coroutine functions

# Langchain Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool

# Import config_manager (singleton)
from config.config_manager import config_manager

# Import user_manager for RBAC checks within services (UserManager class)
from utils.user_manager import UserManager, get_user_tier_capability
from backend.models.user_models import UserProfile

# NEW: Import ApiUsageService for API limit checks and usage tracking
from backend.services.api_usage_service import ApiUsageService

# Import all shared tools (these will be wrapped as Langchain Tools)
from shared_tools.python_interpreter_tool import python_interpreter_with_rbac
from shared_tools.scrapper_tool import scrape_web

from shared_tools.doc_summarizer import summarize_document # This is a direct function

from shared_tools.doc_summarizer import summarize_document

from shared_tools.chart_generation_tool import ChartTools # Import the class
from shared_tools.sentiment_analysis_tool import analyze_sentiment
from shared_tools.query_uploaded_docs_tool import query_uploaded_docs # This is a direct function

# Import the export function from its utility module
from shared_tools.export_utils import export_dataframe_to_file


# Import domain-specific tools (import the classes, not individual functions)

# Import domain-specific tools

from domain_tools.finance_tools.finance_tool import FinanceTools
from domain_tools.crypto_tools.crypto_tool import CryptoTools
from domain_tools.medical_tools.medical_tool import MedicalTools
from domain_tools.news_tools.news_tool import NewsTools
from domain_tools.legal_tools.legal_tool import LegalTools
from domain_tools.education_tools.education_tool import EducationTools
from domain_tools.entertainment_tools.entertainment_tool import EntertainmentTools
from domain_tools.weather_tools import WeatherTools
from domain_tools.travel_tools import TravelTools
from domain_tools.sports_tools import SportsTools
from domain_tools.document_tools import DocumentTools # Import the DocumentTools class

# Import the new dependency functions from backend.dependencies
from backend.dependencies import (
    get_user_manager,
    get_api_usage_service,
    get_firestore_manager,
    get_cloud_storage_utils,
    get_vector_utils_wrapper
)

# Import log_event directly from utils.analytics_tracker
from utils.analytics_tracker import log_event

from domain_tools.weather_tools.weather_tool import get_current_weather, get_weather_forecast
from domain_tools.travel_tools import TravelTools
from domain_tools.sports_tools import SportsTools


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class LLMService:
    """
    Manages interactions with Large Language Models and orchestrates tool usage.
    This service will be called by FastAPI endpoints.
    """
    def __init__(self, 
                 user_manager: UserManager, 
                 api_usage_service: ApiUsageService,
                 firestore_manager: Any, # Use Any for now if the exact class type causes Pydantic issues
                 cloud_storage_utils: Any, # Use Any for now
                 vector_utils_wrapper: Any # Use Any for now
                ):
        """
        Initializes LLMService with necessary dependencies.
        LLM will be loaded dynamically per request.
        """
        self.user_manager = user_manager
        self.api_usage_service = api_usage_service
        self.llm = None # Initialize as None, will be set in chat_with_agent or chat_completion

        # Store injected dependencies for DocumentTools and other tools
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils
        self.vector_utils_wrapper = vector_utils_wrapper

        # Instantiate shared tool classes
        self.chart_tools_instance = ChartTools()
        self.export_dataframe_to_file_func = export_dataframe_to_file # Direct function reference

        # Instantiate domain-specific tool classes
        self.finance_tools = FinanceTools()
        self.crypto_tools = CryptoTools()
        self.medical_tools = MedicalTools()
        self.news_tools = NewsTools()
        self.legal_tools = LegalTools()
        self.education_tools = EducationTools()
        self.entertainment_tools = EntertainmentTools()
        self.weather_tools = WeatherTools()
        self.travel_tools = TravelTools()
        self.sports_tools = SportsTools()
        
        # Instantiate DocumentTools with its specific dependencies
        self.document_tools = DocumentTools(
            vector_utils_wrapper=self.vector_utils_wrapper,
            config_manager=config_manager, # config_manager is a singleton, access directly
            firestore_manager=self.firestore_manager,
            cloud_storage_utils=self.cloud_storage_utils,
            log_event_func=log_event # log_event is a function
        )

        logger.info("LLMService initialized with UserManager, ApiUsageService, ChartTools, ExportUtil, and all domain tools.")

    def _load_llm(self, user_profile: UserProfile, 
                  user_provided_temperature: Optional[float] = None,
                  user_provided_llm_provider: Optional[str] = None,
                  user_provided_model_name: Optional[str] = None):
        """
        Loads the appropriate LLM based on configuration, user's RBAC capabilities,
        and user-provided selections for temperature, provider, and model name.
        """
        user_id = user_profile.user_id
        
        # Determine effective temperature based on RBAC
        can_control_temp = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_temperature_control_enabled', False)
        tier_default_temp = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_default_temperature', config_manager.get('llm.temperature', 0.7))
        max_allowed_temp = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_max_temperature', 1.0)

        effective_temperature = tier_default_temp
        if can_control_temp and user_provided_temperature is not None:
            effective_temperature = min(user_provided_temperature, max_allowed_temp)
            logger.info(f"User {user_id} can control temperature. Using provided {user_provided_temperature}, capped at {max_allowed_temp}. Effective: {effective_temperature}")
        else:
            logger.info(f"User {user_id} cannot control temperature or none provided. Using tier default: {effective_temperature}")

        # Determine effective LLM provider and model name based on RBAC
        can_select_model = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_model_selection_enabled', False)
        
        effective_llm_provider = config_manager.get("llm.provider", "openai")
        effective_model_name = config_manager.get("llm.model_name", "gpt-3.5-turbo")

        if can_select_model:
            if user_provided_llm_provider:
                effective_llm_provider = user_provided_llm_provider
            if user_provided_model_name:
                effective_model_name = user_provided_model_name
            logger.info(f"User {user_id} can select model. Using provided provider '{user_provided_llm_provider}' and model '{user_provided_model_name}'. Effective: {effective_llm_provider}/{effective_model_name}")
        else:
            logger.info(f"User {user_id} cannot select model. Using config defaults: {effective_llm_provider}/{effective_model_name}")

        api_key = None

        if effective_llm_provider == "openai":
            api_key = config_manager.get_secret("openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found in secrets.")
                raise ValueError("OpenAI API key is required for OpenAI LLM provider.")
            
            return ChatOpenAI(model_name=effective_model_name, temperature=effective_temperature, api_key=api_key)
            
        elif effective_llm_provider == "google":
            api_key = config_manager.get_secret("google_api_key")
            if not api_key:
                logger.error("Google API key not found in secrets.")
                raise ValueError("Google API key is required for Google LLM provider.")
            
            # Use ChatGoogleGenerativeAI as imported
            return ChatGoogleGenerativeAI(model=effective_model_name, temperature=effective_temperature, api_key=api_key)
            
        elif effective_llm_provider == "ollama":
            ollama_base_url = config_manager.get("ollama.base_url", "http://localhost:11434")
            logger.info(f"Connecting to Ollama at: {ollama_base_url}")
            return ChatOllama(model=effective_model_name, temperature=effective_temperature, base_url=ollama_base_url)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {effective_llm_provider}")


    def chat_completion(self, messages: List[Dict[str, str]], user_profile: UserProfile,
                        temperature: Optional[float] = None,
                        llm_provider: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Generates a basic chat completion using the configured LLM (without tools).
        """
        try:
            temp_llm = self._load_llm(user_profile=user_profile,
                                      user_provided_temperature=temperature,
                                      user_provided_llm_provider=llm_provider,
                                      user_provided_model_name=model_name)
            
            langchain_messages = [self._convert_to_langchain_message(msg) for msg in messages]
            response = temp_llm.invoke(langchain_messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Error during LLM chat completion for user {user_profile.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM chat completion failed: {e}")

    async def chat_with_agent(self, prompt: str, chat_history: List[Dict[str, str]], user_profile: UserProfile, 
                              user_provided_temperature: Optional[float] = None,
                              user_provided_llm_provider: Optional[str] = None,
                              user_provided_model_name: Optional[str] = None) -> str:
        """
        Orchestrates a chat with an agent, dynamically providing tools based on user's capabilities.
        """
        user_id = user_profile.user_id
        logger.info(f"Agent chat initiated for user: {user_id}, prompt: '{prompt[:100]}...', user_provided_temp: {user_provided_temperature}, user_provided_provider: {user_provided_llm_provider}, user_provided_model: {user_provided_model_name}")

        self.llm = self._load_llm(user_profile, user_provided_temperature, user_provided_llm_provider, user_provided_model_name)

        def get_tool_api_id(tool_func) -> str:
            # Check if the tool_func is a method of a specific tool class instance
            # This allows mapping tool methods to their API IDs.
            if hasattr(tool_func, '__self__'):
                instance = tool_func.__self__
                if isinstance(instance, FinanceTools): return "finance-api-default"
                if isinstance(instance, CryptoTools): return "crypto-api-default"
                if isinstance(instance, MedicalTools): return "medical-api-default"
                if isinstance(instance, NewsTools): return "news-api-default"
                if isinstance(instance, LegalTools): return "legal-api-default"
                if isinstance(instance, EducationTools): return "education-api-default"
                if isinstance(instance, EntertainmentTools): return "entertainment-api-default"
                if isinstance(instance, WeatherTools): return "weather-api-default"
                if isinstance(instance, TravelTools): return "travel-api-default"
                if isinstance(instance, SportsTools): return "sports-api-default"
                if isinstance(instance, DocumentTools): return "document-api"
                if isinstance(instance, ChartTools): return "chart-gen-api"
            
            # Fallback for directly imported functions or special cases
            if tool_func == python_interpreter_with_rbac:
                return "python-interpreter-api"
            if tool_func == scrape_web:
                return "web-scraper-api"
            if tool_func == analyze_sentiment:
                return "sentiment-api"
            
            return "general-tool-api"

        async def wrapped_tool_executor(tool_func, *args, **kwargs):
            api_id = get_tool_api_id(tool_func)
            
            can_proceed = await self.api_usage_service.check_api_limit(user_profile, api_id)
            if not can_proceed:
                logger.warning(f"API limit exceeded for user {user_id}, API {api_id}.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "message": f"API limit exceeded for {api_id} for your tier ({user_profile.tier}). Please upgrade your plan or try again later.",
                        "code": "API_LIMIT_EXCEEDED"
                    }
                )
            
            # Pass user_token to tools that need it for internal RBAC/logging
            tool_args = kwargs.copy()
            if 'user_token' not in tool_args and hasattr(user_profile, 'user_id'):
                tool_args['user_token'] = user_profile.user_id
            elif 'user_token' not in tool_args:
                tool_args['user_token'] = "default" # Fallback for testing or anonymous

            # Special handling for python_interpreter_with_rbac and generate_and_save_chart
            # to pass their specific dependencies
            if tool_func == python_interpreter_with_rbac:
                tool_args['chart_tools'] = self.chart_tools_instance
                tool_args['export_dataframe_to_file_func'] = self.export_dataframe_to_file_func
            
            logger.debug(f"Executing tool {tool_func.__name__} for user {user_id} (API: {api_id})...")
            
            # Correctly unpack args and kwargs for the tool_func call
            tool_output = await tool_func(*args, **tool_args)
            
            await self.api_usage_service.increment_api_usage(user_id, api_id)
            logger.debug(f"Tool {tool_func.__name__} executed successfully. Usage incremented.")
            return tool_output

        available_tools = []

        # Shared Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'web_search_enabled', False):
            available_tools.append(Tool(
                name="scrape_web",
                func=lambda query: wrapped_tool_executor(scrape_web, query),
                description="A tool to perform web searches and scrape content from URLs. Input should be a search query string."
            ))
            logger.debug(f"Tool 'scrape_web' added for user {user_id}")
        
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'data_analysis_enabled', False):
            available_tools.append(Tool(
                name="python_interpreter_with_rbac",
                func=lambda code: wrapped_tool_executor(
                    python_interpreter_with_rbac,
                    code,
                    # chart_tools and export_dataframe_to_file_func are passed within wrapped_tool_executor
                ),
                description="A powerful Python interpreter for data analysis, complex calculations, time series analysis, regression analysis, or any machine learning tasks. Input should be valid Python code. This tool also provides access to `chart_tools` for charting and `export_data_to_file` for exporting dataframes."
            ))
            logger.debug(f"Tool 'python_interpreter_with_rbac' added for user {user_id}")
        
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'chart_generation_enabled', False):
            available_tools.append(Tool(
                name="generate_and_save_chart",
                func=lambda data_json, chart_type, x_column=None, y_column=None, color_column=None, title="Generated Chart", x_label=None, y_label=None, library="matplotlib", export_format="png": wrapped_tool_executor(
                    self.chart_tools_instance.generate_and_save_chart,
                    data_json=data_json,
                    chart_type=chart_type,
                    x_column=x_column,
                    y_column=y_column,
                    color_column=color_column,
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    library=library,
                    export_format=export_format
                ),
                description="Generates and saves a chart (e.g., line, bar, scatter, pie, histogram, boxplot) from provided JSON data. Input should be a JSON string of data, chart type, and optional columns for x, y, color, title, and labels. Supported libraries are matplotlib, seaborn, plotly. Supported export formats are png, jpeg, svg, html (for plotly)."
            ))
            logger.debug(f"Tool 'generate_and_save_chart' added for user {user_id}")

        if self.user_manager.get_user_tier_capability(user_profile.tier, 'sentiment_analysis_enabled', False):
            available_tools.append(Tool(
                name="analyze_sentiment",
                func=lambda text: wrapped_tool_executor(analyze_sentiment, text),
                description="Analyzes the sentiment of a given text. Input should be a string of text."
            ))
            logger.debug(f"Tool 'analyze_sentiment' added for user {user_id}")
        
        # Document Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'document_upload_enabled', False):
            available_tools.append(Tool(
                name="document_process_uploaded_document",
                func=lambda file_name, file_content_base64: wrapped_tool_executor(self.document_tools.document_process_uploaded_document, file_name=file_name, file_content_base64=file_content_base64),
                description="Uploads a document to cloud storage and processes it for vector indexing, making it searchable. Input: file_name (str), file_content_base64 (str)."
            ))
            logger.debug(f"Tool 'document_process_uploaded_document' added for user {user_id}")

        if self.user_manager.get_user_tier_capability(user_profile.tier, 'document_query_enabled', False):
            available_tools.append(Tool(
                name="document_query_uploaded_docs",
                func=lambda query_text, section="general", export=False, k=5: wrapped_tool_executor(self.document_tools.document_query_uploaded_docs, query=query_text, section=section, export=export, k=k),
                description="Queries previously uploaded and indexed documents for a user using vector similarity search. Input should be a query string, an optional section (e.g., 'general', 'financial'), an optional boolean for export, and an optional number of results (k)."
            ))
            logger.debug(f"Tool 'document_query_uploaded_docs' added for user {user_id}")

        if self.user_manager.get_user_tier_capability(user_profile.tier, 'summarization_enabled', False):
            available_tools.append(Tool(
                name="document_summarize_document_by_path",
                func=lambda file_path_str, user_token="default": wrapped_tool_executor(self.document_tools.document_summarize_document_by_path, file_path_str=file_path_str, user_token=user_token),
                description="Summarizes a document located at the given file path. The file path should be accessible by the system. Input should be the full file path string."
            ))
            logger.debug(f"Tool 'document_summarize_document_by_path' added for user {user_id}")

        if self.user_manager.get_user_tier_capability(user_profile.tier, 'web_search_enabled', False): # Assuming web search is used for document_search_web
            available_tools.append(Tool(
                name="document_search_web",
                func=lambda query: wrapped_tool_executor(self.document_tools.document_search_web, query=query),
                description="Searches the web for general document-related information. Input should be a search query string."
            ))
            logger.debug(f"Tool 'document_search_web' added for user {user_id}")


        # Domain-specific Tools - Finance Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'finance_tool_access', False):
            available_tools.extend([
                Tool(name="get_stock_price", func=lambda symbol: wrapped_tool_executor(self.finance_tools.get_stock_price, symbol), description="Retrieves the current stock price for a given stock symbol. Input should be a stock symbol (e.g., 'AAPL')."),
                Tool(name="get_company_news", func=lambda symbol, from_date, to_date: wrapped_tool_executor(self.finance_tools.get_company_news, symbol, from_date, to_date), description="Fetches recent news for a company by its stock symbol within a date range. Input: symbol (str), from_date (YYYY-MM-DD), to_date (YYYY-MM-DD)."),
                Tool(name="lookup_stock_symbol", func=lambda company_name: wrapped_tool_executor(self.finance_tools.lookup_stock_symbol, company_name), description="Looks up the stock symbol for a given company name. Input: company_name (str).")
            ])
            logger.debug(f"Finance tools (current price, company news, symbol lookup) added for user {user_id}")
        
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'historical_data_access', False):
            available_tools.append(Tool(
                name="get_historical_stock_prices",
                func=lambda symbol, start_date, end_date: wrapped_tool_executor(self.finance_tools.get_historical_stock_prices, symbol, start_date, end_date),
                description="Retrieves historical stock prices for a given symbol and date range. Input: symbol (str), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)."
            ))
            logger.debug(f"Tool 'get_historical_stock_prices' added for user {user_id}")

        # Crypto Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'crypto_tool_access', False):
            available_tools.extend([
                Tool(name="get_crypto_price", func=lambda coin_id: wrapped_tool_executor(self.crypto_tools.get_crypto_price, coin_id), description="Retrieves the current price of a cryptocurrency by its ID."),
                Tool(name="get_historical_crypto_prices", func=lambda coin_id, vs_currency, days: wrapped_tool_executor(self.crypto_tools.get_historical_crypto_prices, coin_id, vs_currency, days), description="Retrieves historical prices for a cryptocurrency."),
                Tool(name="get_crypto_id_by_symbol", func=lambda symbol: wrapped_tool_executor(self.crypto_tools.get_crypto_id_by_symbol, symbol), description="Looks up the cryptocurrency ID by its symbol.")
            ])
            logger.debug(f"Crypto tools added for user {user_id}")

        # Medical Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'medical_tool_access', False):
            available_tools.extend([
                Tool(name="get_drug_info", func=lambda drug_name: wrapped_tool_executor(self.medical_tools.get_drug_info, drug_name), description="Retrieves information about a specific drug."),
                Tool(name="get_symptom_info", func=lambda symptom_name: wrapped_tool_executor(self.medical_tools.get_symptom_info, symptom_name), description="Retrieves information about a specific symptom.")
            ])
            logger.debug(f"Medical tools added for user {user_id}")

        # News Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'news_tool_access', False):
            available_tools.append(Tool(name="get_general_news", func=lambda query: wrapped_tool_executor(self.news_tools.get_general_news, query), description="Fetches general news articles based on a query."))
            logger.debug(f"General news tool added for user {user_id}")
        
        # Legal Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'legal_tool_access', False):
            available_tools.extend([
                Tool(name="get_legal_definition", func=lambda term: wrapped_tool_executor(self.legal_tools.get_legal_definition, term), description="Retrieves the definition of a legal term."),
                Tool(name="get_case_summary", func=lambda case_name: wrapped_tool_executor(self.legal_tools.get_case_summary, case_name), description="Retrieves a summary of a legal case.")
            ])
            logger.debug(f"Legal tools (definition, case summary) added for user {user_id}")
        
        # Education Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'education_tool_access', False):
            available_tools.extend([
                Tool(name="get_academic_definition", func=lambda term: wrapped_tool_executor(self.education_tools.get_academic_definition, term), description="Retrieves the definition of an academic term."),
                Tool(name="get_historical_event_summary", func=lambda event_name: wrapped_tool_executor(self.education_tools.get_historical_event_summary, event_name), description="Retrieves a summary of a historical event.")
            ])
            logger.debug(f"Education tools (academic definition, historical event summary) added for user {user_id}")
        
        # Entertainment Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'entertainment_tool_access', False):
            available_tools.extend([
                Tool(name="get_movie_details", func=lambda movie_title: wrapped_tool_executor(self.entertainment_tools.get_movie_details, movie_title), description="Retrieves details about a movie."),
                Tool(name="get_music_artist_info", func=lambda artist_name: wrapped_tool_executor(self.entertainment_tools.get_music_artist_info, artist_name), description="Retrieves information about a music artist.")
            ])
            logger.debug(f"Entertainment tools (movie details, music artist info) added for user {user_id}")

        # Weather Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'weather_tool_access', False):
            available_tools.extend([
                Tool(name="get_current_weather", func=lambda location, unit="celsius": wrapped_tool_executor(self.weather_tools.get_current_weather, location, unit=unit), description="Retrieves current weather conditions for a location. Input: location (str), optional unit (str, 'celsius' or 'fahrenheit')."),
                Tool(name="get_weather_forecast", func=lambda location, days=3, unit="celsius": wrapped_tool_executor(self.weather_tools.get_weather_forecast, location, days=days, unit=unit), description="Retrieves weather forecast for a location for a number of days (max 10). Input: location (str), optional days (int), optional unit (str, 'celsius' or 'fahrenheit')."),
                Tool(name="get_air_quality", func=lambda location: wrapped_tool_executor(self.weather_tools.get_air_quality, location), description="Retrieves air quality information for a specified location. Input: location (str).")
            ])
            logger.debug(f"Weather tools (current weather, forecast, air quality) added for user {user_id}")
        
        # Travel Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'travel_tool_access', False):
            available_tools.extend([
                Tool(name="search_flights", func=lambda origin, destination, departure_date, return_date=None, adults=1, currency="USD": wrapped_tool_executor(self.travel_tools.search_flights, origin, destination, departure_date, return_date=return_date, adults=adults, currency=currency), description="Searches for flights between specified origin and destination airports for given dates. Uses IATA airport codes (e.g., 'JFK', 'LAX', 'LHR', 'CDG'). Departure and return dates should be in YYYY-MM-DD format."),
                Tool(name="search_hotels", func=lambda city_code, check_in_date, check_out_date, adults=1: wrapped_tool_executor(self.travel_tools.search_hotels, city_code, check_in_date, check_out_date, adults=adults), description="Searches for hotels in a specified city for given check-in and check-out dates. Uses IATA city codes (e.g., 'PAR' for Paris, 'NYC' for New York). Dates should be in YYYY-MM-DD format."),
                Tool(name="get_destination_info", func=lambda destination_name: wrapped_tool_executor(self.travel_tools.get_destination_info, destination_name), description="Retrieves general information about a travel destination, including description, best time to visit, currency, and language.")
            ])
            logger.debug(f"Travel tools (search flights, search hotels, get destination info) added for user {user_id}")
        
        # Sports Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'sports_tool_access', False):
            available_tools.extend([
                Tool(name="get_latest_scores", func=lambda sport=None, team=None: wrapped_tool_executor(self.sports_tools.get_latest_scores, sport=sport, team=team), description="Retrieves the latest scores for sports matches, optionally filtered by sport or team."),
                Tool(name="get_upcoming_events", func=lambda sport=None: wrapped_tool_executor(self.sports_tools.get_upcoming_events, sport=sport), description="Retrieves upcoming sports events, optionally filtered by sport."),
                Tool(name="get_player_stats", func=lambda player_name, sport=None: wrapped_tool_executor(self.sports_tools.get_player_stats, player_name, sport=sport), description="Retrieves statistics for a specified player, optionally filtered by sport."),
                Tool(name="get_team_stats", func=lambda team_name, sport=None: wrapped_tool_executor(self.sports_tools.get_team_stats, team_name, sport=sport), description="Retrieves statistics for a specified team, optionally filtered by sport."),
                Tool(name="get_league_info", func=lambda league_name: wrapped_tool_executor(self.sports_tools.get_league_info, league_name), description="Retrieves general information about a specified sports league.")
            ])
            logger.debug(f"Sports tools added for user {user_id}")


        if not available_tools:
            logger.info(f"No specialized tools available for user {user_id}. Falling back to chat completion.")
            return await self.chat_completion(chat_history + [{"role": "user", "content": prompt}],
                                        user_profile=user_profile,
                                        temperature=user_provided_temperature,
                                        llm_provider=user_provided_llm_provider,
                                        model_name=user_provided_model_name)

        # Convert chat history to Langchain BaseMessage format
        langchain_chat_history = [self._convert_to_langchain_message(msg) for msg in chat_history]

        # Define the prompt template for the agent
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a helpful AI assistant with access to various tools. "
                "Use the tools to answer questions and fulfill requests. "
                "For web search, use `scrape_web`. "
                "For sentiment analysis, use `analyze_sentiment`. "
                "For current stock prices, use `get_stock_price`. "
                "For historical stock prices, use `get_historical_stock_prices`. "
                "For company news, use `get_company_news`. "
                "To find a stock symbol from a company name, use `lookup_stock_symbol`. "
                "For current cryptocurrency prices, use `get_crypto_price`. "
                "For historical cryptocurrency prices, use `get_historical_crypto_prices`. "
                "To find a cryptocurrency ID from its symbol, use `get_crypto_id_by_symbol`. "
                "For drug information, use `get_drug_info`. "
                "For symptom information, use `get_symptom_info`. "
                "For general news, use `get_general_news`. "
                "For legal term definitions, use `get_legal_definition`. "
                "For legal case summaries, use `get_case_summary`. "
                "For academic term definitions, use `get_academic_definition`. "
                "For historical event summaries, use `get_historical_event_summary`. "
                "For movie details, use `get_movie_details`. "
                "For music artist information, use `get_music_artist_info`. "
                "For current weather, use `get_current_weather`. "
                "For weather forecasts, use `get_weather_forecast`. "
                "For air quality information, use `get_air_quality`. "
                "For finding flights, use `search_flights`. "
                "For finding hotels, use `search_hotels`. "
                "For destination information, use `get_destination_info`. "
                "For player statistics (e.g., career stats, trophies, rings), use `get_player_stats`. "
                "For team or club statistics (e.g., season stats, major trophies, standings), use `get_team_stats`. "
                "For sports league information (e.g., champions, top scorers), use `get_league_info`. "
                "For latest sports scores, use `get_latest_scores`. "
                "For upcoming sports events, use `get_upcoming_events`. "
                "For **uploading and processing** new documents for search, use `document_process_uploaded_document`. "
                "For **querying** uploaded documents, use `document_query_uploaded_docs`. "
                "For **summarizing** documents by path, use `document_summarize_document_by_path`. "
                "For **web search related to documents**, use `document_search_web`. "
                "For **data analysis**, complex calculations, time series analysis, regression analysis, "
                "or any other machine learning tasks (supervised or unsupervised), use the `python_interpreter_with_rbac` tool. "
                "For generating charts from data, use `generate_and_save_chart`. "
                "Always provide comprehensive answers based on tool outputs. "
                "If a tool call fails, inform the user and try to explain why or suggest alternatives."
                "When providing historical data, if asked to plot, use `generate_and_save_chart` with the JSON output from `get_historical_stock_prices` or `get_historical_crypto_prices`."
                "When analyzing data from uploaded documents, use `document_query_uploaded_docs` first, then pass the relevant content to `python_interpreter_with_rbac` for analysis."
                "Remember to pass the `user_token` to any tool that requires it for RBAC or logging. The `wrapped_tool_executor` handles this automatically."
                "If a user asks for a stock by name (e.g., 'Apple'), first use `lookup_stock_symbol` to get the ticker, then use the appropriate stock tool."
                "If a user asks for crypto by symbol (e.g., 'btc'), first use `get_crypto_id_by_symbol` to get the ID, then use the appropriate crypto tool."
                "If you need to export data, you can do so by calling the `export_data_to_file` function *from within* the `python_interpreter_with_rbac` tool."
            ),
            *langchain_chat_history,
            HumanMessage(content="{input}"),
            AIMessage(content="{agent_scratchpad}"),
        ])

        # Create the Langchain agent
        agent = create_react_agent(self.llm, available_tools, prompt_template)
        # AgentExecutor will handle parsing errors and verbose logging
        agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True)
        
        logger.info("Using real Langchain AgentExecutor.")

        try:
            # Invoke the agent with the current prompt and chat history
            response = await agent_executor.invoke({
                "input": prompt,
                "chat_history": langchain_chat_history,
                "user_profile": user_profile # Pass user_profile to the agent
            })
            return response["output"]
        except HTTPException as e:
            raise
        except Exception as e:
            logger.error(f"Error during Langchain agent invocation for user {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent execution failed: {str(e)}")

    def _convert_to_langchain_message(self, message: Dict[str, str]) -> BaseMessage:
        """Helper to convert dictionary messages to Langchain BaseMessage objects."""
        if message["role"] == "user":
            return HumanMessage(content=message["content"])
        elif message["role"] == "assistant":
            return AIMessage(content=message["content"])
        elif message["role"] == "system":
            return SystemMessage(content=message["content"])
        else:
            raise ValueError(f"Unknown message role: {message['role']}")

# Dependency function for FastAPI
async def get_llm_service_dependency(
    user_manager_instance: UserManager = Depends(get_user_manager),
    api_usage_service_instance: ApiUsageService = Depends(get_api_usage_service),
    firestore_manager_instance: Any = Depends(get_firestore_manager),
    cloud_storage_utils_instance: Any = Depends(get_cloud_storage_utils),
    vector_utils_wrapper_instance: Any = Depends(get_vector_utils_wrapper)
) -> LLMService:
    """
    FastAPI dependency that provides an LLMService instance.
    This ensures that the LLMService is correctly initialized with its dependencies.
    """
    return LLMService(
        user_manager=user_manager_instance,
        api_usage_service=api_usage_service_instance,
        firestore_manager=firestore_manager_instance,
        cloud_storage_utils=cloud_storage_utils_instance,
        vector_utils_wrapper=vector_utils_wrapper_instance
    )

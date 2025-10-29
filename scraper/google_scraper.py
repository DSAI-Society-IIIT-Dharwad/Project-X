import requests
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()


class GoogleNewsScraper:
    """
    A scraper for Google News using the NewsAPI.org service.
    You'll need to sign up for a free API key at https://newsapi.org/
    """
    
    def __init__(self, api_key: str, output_dir: str = r"..\data\raw"):
        """
        Initialize the scraper with API key and output directory.
        
        Args:
            api_key: Your NewsAPI.org API key
            output_dir: Directory to save JSON files
        """
        self.api_key = os.getenv("NEWS_API_KEY")
        self.output_dir = output_dir
        self.base_url = "https://newsapi.org/v2"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_top_headlines(self, 
                           country: str = 'us', 
                           category: Optional[str] = None,
                           query: Optional[str] = None,
                           page_size: int = 100) -> Dict:
        """
        Fetch top headlines from Google News.
        
        Args:
            country: Country code (e.g., 'us', 'gb', 'in')
            category: Category (business, entertainment, general, health, science, sports, technology)
            query: Keywords to search for
            page_size: Number of results (max 100)
            
        Returns:
            Dictionary containing news articles
        """
        endpoint = f"{self.base_url}/top-headlines"
        
        params = {
            'apiKey': self.api_key,
            'pageSize': page_size,
            'country': country
        }
        
        if category:
            params['category'] = category
        if query:
            params['q'] = query
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching headlines: {e}")
            return {}
    
    def fetch_everything(self, 
                        query: str,
                        from_date: Optional[str] = None,
                        to_date: Optional[str] = None,
                        language: str = 'en',
                        sort_by: str = 'publishedAt',
                        page_size: int = 100) -> Dict:
        """
        Search all articles from Google News.
        
        Args:
            query: Keywords or phrases to search for
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            language: Language code (e.g., 'en', 'es', 'fr')
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of results (max 100)
            
        Returns:
            Dictionary containing news articles
        """
        endpoint = f"{self.base_url}/everything"
        
        params = {
            'apiKey': self.api_key,
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles: {e}")
            return {}
    
    def save_to_json(self, data: Dict, filename: Optional[str] = None) -> str:
        """
        Save scraped data to JSON file.
        
        Args:
            data: Dictionary containing news data
            filename: Custom filename (without extension)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_data_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Data saved successfully to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving file: {e}")
            return ""
    
    def scrape_and_save(self, 
                       scrape_type: str = 'headlines',
                       **kwargs) -> str:
        """
        Scrape news and save to JSON in one step.
        
        Args:
            scrape_type: Type of scrape ('headlines' or 'everything')
            **kwargs: Additional arguments for the scraping method
            
        Returns:
            Path to saved file
        """
        if scrape_type == 'headlines':
            data = self.fetch_top_headlines(**kwargs)
        elif scrape_type == 'everything':
            data = self.fetch_everything(**kwargs)
        else:
            print("Invalid scrape_type. Use 'headlines' or 'everything'")
            return ""
        
        if data.get('status') == 'ok':
            print(f"Successfully fetched {data.get('totalResults', 0)} articles")
            return self.save_to_json(data)
        else:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return ""


# Example usage
if __name__ == "__main__":
    # Replace with your actual NewsAPI.org API key
    API_KEY = "YOUR_API_KEY_HERE"
    
    # Initialize scraper
    scraper = GoogleNewsScraper(api_key=API_KEY)
    
    # Example 1: Scrape top headlines from US
    print("Scraping top headlines from US...")
    scraper.scrape_and_save(
        scrape_type='headlines',
        country='us',
        page_size=50
    )
    
    # Example 2: Scrape technology news
    print("\nScraping technology news...")
    scraper.scrape_and_save(
        scrape_type='headlines',
        country='us',
        category='technology',
        page_size=50
    )
    
    # Example 3: Search for specific topic
    print("\nSearching for 'artificial intelligence' news...")
    scraper.scrape_and_save(
        scrape_type='everything',
        query='artificial intelligence',
        language='en',
        sort_by='publishedAt',
        page_size=50
    )
    
    # Example 4: Scrape recent news (last 24-48 hours - free tier limitation)
    print("\nSearching recent technology news...")
    scraper.scrape_and_save(
        scrape_type='everything',
        query='technology',
        
        page_size=50
    )
    
    print("\nAll scraping completed!")
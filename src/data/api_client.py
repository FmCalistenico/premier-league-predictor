"""
API clients for fetching football data from external sources.
Includes rate limiting, retry logic, and error handling.
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils import LoggerMixin, Config


class APIFootballClient(LoggerMixin):
    """
    Client for API-Football (RapidAPI).

    Features:
    - Rate limiting (10 requests/minute)
    - Exponential backoff retry logic
    - Automatic header management
    - Response validation

    API Documentation: https://www.api-football.com/documentation-v3
    """

    BASE_URL = "https://v3.football.api-sports.io"
    RATE_LIMIT = 10  # requests per minute
    RATE_LIMIT_WINDOW = 60  # seconds

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API-Football client.

        Args:
            api_key: RapidAPI key. If None, reads from environment/config.
        """
        self.config = Config()
        self.api_key = api_key or self.config.api_keys.get('football_data', '')

        if not self.api_key:
            self.logger.warning("API key not provided. API calls will fail.")

        # Rate limiting tracking
        self.request_times: List[float] = []

        # Setup session with retry logic
        self.session = self._create_session()

        self.logger.info("APIFootballClient initialized")

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()

        # Retry strategy with exponential backoff
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,  # 1, 2, 4, 8, 16 seconds
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        return {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': self.api_key,
            'x-apisports-key': self.api_key,  # Alternative header name
        }

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting (10 requests per minute)."""
        current_time = time.time()

        # Remove requests older than the rate limit window
        self.request_times = [
            t for t in self.request_times
            if current_time - t < self.RATE_LIMIT_WINDOW
        ]

        # If we've hit the rate limit, wait
        if len(self.request_times) >= self.RATE_LIMIT:
            oldest_request = self.request_times[0]
            wait_time = self.RATE_LIMIT_WINDOW - (current_time - oldest_request)

            if wait_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s")
                time.sleep(wait_time + 0.1)  # Add small buffer

        # Record this request
        self.request_times.append(time.time())

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting and error handling.

        Args:
            endpoint: API endpoint (e.g., '/fixtures')
            params: Query parameters

        Returns:
            API response as dictionary

        Raises:
            requests.RequestException: If request fails after retries
        """
        self._enforce_rate_limit()

        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers()

        self.logger.debug(f"Making request to {endpoint} with params: {params}")

        try:
            response = self.session.get(
                url,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if 'errors' in data and data['errors']:
                error_msg = data['errors']
                self.logger.error(f"API returned errors: {error_msg}")
                raise requests.RequestException(f"API errors: {error_msg}")

            if 'response' not in data:
                self.logger.error(f"Invalid API response structure: {data}")
                raise requests.RequestException("Invalid response structure")

            self.logger.debug(f"Request successful. Got {len(data['response'])} results")
            return data

        except requests.Timeout:
            self.logger.error(f"Request timeout for {endpoint}")
            raise
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {endpoint}: {str(e)}")
            raise

    def get_fixtures(
        self,
        league_id: Optional[int] = None,
        season: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        team_id: Optional[int] = None,
        fixture_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get fixtures/matches data.

        Args:
            league_id: League ID (e.g., 39 for Premier League)
            season: Season year (e.g., '2024')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            team_id: Filter by team ID
            fixture_id: Get specific fixture

        Returns:
            List of fixture dictionaries

        Example:
            >>> client = APIFootballClient()
            >>> fixtures = client.get_fixtures(league_id=39, season='2024')
        """
        params = {}

        if league_id:
            params['league'] = league_id
        if season:
            params['season'] = season
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if team_id:
            params['team'] = team_id
        if fixture_id:
            params['id'] = fixture_id

        self.logger.info(f"Fetching fixtures with params: {params}")

        try:
            response = self._make_request('/fixtures', params)
            fixtures = response['response']

            self.logger.info(f"Successfully fetched {len(fixtures)} fixtures")
            return fixtures

        except Exception as e:
            self.logger.error(f"Failed to fetch fixtures: {str(e)}")
            raise

    def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: str
    ) -> Dict[str, Any]:
        """
        Get team statistics for a season.

        Args:
            team_id: Team ID
            league_id: League ID
            season: Season year (e.g., '2024')

        Returns:
            Team statistics dictionary

        Example:
            >>> stats = client.get_team_statistics(
            ...     team_id=33,  # Manchester United
            ...     league_id=39,
            ...     season='2024'
            ... )
        """
        params = {
            'team': team_id,
            'league': league_id,
            'season': season
        }

        self.logger.info(f"Fetching team statistics for team {team_id}, season {season}")

        try:
            response = self._make_request('/teams/statistics', params)
            stats = response['response']

            self.logger.info(f"Successfully fetched statistics for team {team_id}")
            return stats

        except Exception as e:
            self.logger.error(f"Failed to fetch team statistics: {str(e)}")
            raise

    def get_standings(
        self,
        league_id: int,
        season: str
    ) -> List[Dict[str, Any]]:
        """
        Get league standings.

        Args:
            league_id: League ID (e.g., 39 for Premier League)
            season: Season year (e.g., '2024')

        Returns:
            List of standings dictionaries

        Example:
            >>> standings = client.get_standings(league_id=39, season='2024')
        """
        params = {
            'league': league_id,
            'season': season
        }

        self.logger.info(f"Fetching standings for league {league_id}, season {season}")

        try:
            response = self._make_request('/standings', params)
            standings = response['response']

            self.logger.info(f"Successfully fetched standings")
            return standings

        except Exception as e:
            self.logger.error(f"Failed to fetch standings: {str(e)}")
            raise

    def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        last: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get head-to-head matches between two teams.

        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last: Number of last matches to retrieve

        Returns:
            List of head-to-head fixtures
        """
        params = {
            'h2h': f"{team1_id}-{team2_id}"
        }

        if last:
            params['last'] = last

        self.logger.info(f"Fetching head-to-head: {team1_id} vs {team2_id}")

        try:
            response = self._make_request('/fixtures/headtohead', params)
            h2h = response['response']

            self.logger.info(f"Successfully fetched {len(h2h)} head-to-head matches")
            return h2h

        except Exception as e:
            self.logger.error(f"Failed to fetch head-to-head: {str(e)}")
            raise


class FootballDataCSVClient(LoggerMixin):
    """
    Client for downloading CSV data from football-data.co.uk.

    Provides historical match data in CSV format.
    Website: https://www.football-data.co.uk
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    # Season code mapping (2-digit year format)
    SEASON_CODES = {
        '2324': '2324',  # 2023-2024
        '2425': '2425',  # 2024-2025
        '2526': '2526',  # 2025-2026
        '2223': '2223',  # 2022-2023
        '2122': '2122',  # 2021-2022
        '2021': '2021',  # 2020-2021
    }

    def __init__(self):
        """Initialize Football Data CSV client."""
        self.session = requests.Session()
        self.logger.info("FootballDataCSVClient initialized")

    def _build_url(self, season: str, division: str = "E0") -> str:
        """
        Build URL for CSV download.

        Args:
            season: Season code (e.g., '2425' for 2024-2025)
            division: Division code (E0=Premier League, E1=Championship, etc.)

        Returns:
            Full URL to CSV file
        """
        return f"{self.BASE_URL}/{season}/{division}.csv"

    def get_season_data(
        self,
        season: str = "2425",
        division: str = "E0"
    ) -> pd.DataFrame:
        """
        Download and parse CSV data for a single season.

        Args:
            season: Season code (e.g., '2425' for 2024-2025)
            division: Division code (E0=Premier League)

        Returns:
            DataFrame with season data

        Raises:
            requests.RequestException: If download fails

        Example:
            >>> client = FootballDataCSVClient()
            >>> df = client.get_season_data(season='2425')
        """
        url = self._build_url(season, division)

        self.logger.info(f"Downloading CSV from {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            # Add metadata
            df['season'] = season
            df['source'] = 'football-data.co.uk'
            df['download_date'] = datetime.now().strftime('%Y-%m-%d')

            self.logger.info(f"Successfully downloaded {len(df)} matches for season {season}")
            return df

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.error(f"CSV not found for season {season} at {url}")
                raise ValueError(f"Season {season} not available")
            else:
                self.logger.error(f"HTTP error downloading CSV: {str(e)}")
                raise
        except Exception as e:
            self.logger.error(f"Failed to download CSV: {str(e)}")
            raise

    def get_multiple_seasons(
        self,
        seasons: List[str],
        division: str = "E0"
    ) -> pd.DataFrame:
        """
        Download data for multiple seasons and combine.

        Args:
            seasons: List of season codes (e.g., ['2324', '2425'])
            division: Division code (E0=Premier League)

        Returns:
            Combined DataFrame with all seasons

        Example:
            >>> client = FootballDataCSVClient()
            >>> df = client.get_multiple_seasons(['2223', '2324', '2425'])
        """
        self.logger.info(f"Downloading data for {len(seasons)} seasons")

        all_data = []
        failed_seasons = []

        for season in seasons:
            try:
                df = self.get_season_data(season, division)
                all_data.append(df)

                # Small delay between requests to be polite
                time.sleep(0.5)

            except Exception as e:
                self.logger.warning(f"Failed to download season {season}: {str(e)}")
                failed_seasons.append(season)

        if not all_data:
            raise ValueError("Failed to download data for all seasons")

        # Combine all seasons
        combined_df = pd.concat(all_data, ignore_index=True)

        self.logger.info(
            f"Successfully combined {len(all_data)} seasons. "
            f"Total {len(combined_df)} matches. "
            f"Failed seasons: {failed_seasons if failed_seasons else 'None'}"
        )

        return combined_df

    def get_available_seasons(self) -> List[str]:
        """
        Get list of available season codes.

        Returns:
            List of season codes
        """
        return list(self.SEASON_CODES.keys())

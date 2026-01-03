# Data Module

This module handles data extraction from multiple sources for Premier League match prediction.

## Components

### 1. API Clients (`api_client.py`)

#### APIFootballClient
Connects to API-Football (RapidAPI) with the following features:
- **Rate limiting**: 10 requests per minute
- **Retry logic**: Exponential backoff (1, 2, 4, 8, 16 seconds)
- **Methods**:
  - `get_fixtures()` - Retrieve match fixtures
  - `get_team_statistics()` - Get team stats for a season
  - `get_standings()` - Get league standings
  - `get_head_to_head()` - Get H2H matches between teams

**Example:**
```python
from src.data import APIFootballClient

client = APIFootballClient()
fixtures = client.get_fixtures(league_id=39, season='2024')
standings = client.get_standings(league_id=39, season='2024')
```

#### FootballDataCSVClient
Downloads historical data from football-data.co.uk:
- **Methods**:
  - `get_season_data(season)` - Download single season CSV
  - `get_multiple_seasons(seasons)` - Download and combine multiple seasons
  - `get_available_seasons()` - List available season codes

**Example:**
```python
from src.data import FootballDataCSVClient

client = FootballDataCSVClient()
df = client.get_season_data(season='2425')  # 2024-2025 season
df_multi = client.get_multiple_seasons(['2223', '2324', '2425'])
```

### 2. Data Extractor (`extractor.py`)

#### DataExtractor
Orchestrates extraction from multiple sources:
- **Methods**:
  - `extract_from_api()` - Extract from API-Football
  - `extract_from_csv()` - Extract from CSV files
  - `save_raw_data()` - Save with versioning (YYYY-MM-DD_HHMMSS)
  - `run_extraction()` - Complete extraction pipeline

**Features:**
- Automatic versioning by timestamp
- Metadata tracking (JSON)
- Error recovery
- Multi-source support

**Example:**
```python
from src.data import DataExtractor

extractor = DataExtractor()

# Extract from CSV only
results = extractor.run_extraction(
    sources=['csv'],
    csv_seasons=['2223', '2324', '2425'],
    save_data=True
)

# Extract from both sources
results = extractor.run_extraction(
    sources=['csv', 'api'],
    league_id=39,
    season='2024',
    csv_seasons=['2324', '2425']
)
```

## Data Structure

### Raw Data Directory Structure
```
data/raw/
├── csv/
│   ├── csv_data_2025-01-02_143022.csv
│   ├── csv_data_2025-01-02_143022_metadata.json
│   └── ...
├── api/
│   ├── api_data_2025-01-02_144530.json
│   ├── api_data_2025-01-02_144530_metadata.json
│   └── ...
└── extraction_summary_20250102_143022.json
```

### Metadata Format
Each extraction includes metadata:
```json
{
  "source": "csv",
  "seasons": ["2223", "2324", "2425"],
  "extraction_date": "2025-01-02T14:30:22",
  "matches_count": 1140,
  "extraction_status": "success",
  "extractor_version": "1.0.0"
}
```

## Usage

### Command Line Script
```bash
# Extract CSV data for multiple seasons
python scripts/extract_data.py --sources csv --seasons 2223 2324 2425

# Extract from API
python scripts/extract_data.py --sources api --league-id 39 --season 2024

# Extract from all sources
python scripts/extract_data.py --sources all --seasons 2324 2425

# Dry run (don't save)
python scripts/extract_data.py --sources csv --no-save --verbose
```

### Programmatic Usage
```python
from src.utils import setup_logging
from src.data import DataExtractor

# Setup logging
setup_logging()

# Initialize extractor
extractor = DataExtractor()

# Run extraction
results = extractor.run_extraction(
    sources=['csv', 'api'],
    csv_seasons=['2223', '2324', '2425'],
    league_id=39,
    season='2024',
    save_data=True
)

# Check results
print(f"Extraction ID: {results['extraction_id']}")
print(f"Files saved: {results['files_saved']}")
```

## Configuration

### Environment Variables (.env)
```env
# API Keys
FOOTBALL_DATA_API_KEY=your_rapidapi_key_here

# League Settings
LEAGUE_ID=39
CURRENT_SEASON=2024-2025
```

### Config File (config/config.yaml)
```yaml
data:
  sources:
    - football_data_api
    - csv_files

paths:
  data:
    raw: ./data/raw
```

## Season Codes

CSV season codes (2-digit year format):
- `2425` = 2024-2025
- `2324` = 2023-2024
- `2223` = 2022-2023
- `2122` = 2021-2022

API season codes (year format):
- `2024` = 2024-2025 season
- `2023` = 2023-2024 season

## Error Handling

All clients include comprehensive error handling:
- **Rate limiting**: Automatic waiting when limit reached
- **Retry logic**: Exponential backoff for transient failures
- **Validation**: Response structure validation
- **Logging**: Detailed error logging

## Rate Limits

### API-Football
- **Free tier**: 100 requests/day
- **Client limit**: 10 requests/minute
- **Automatic**: Rate limiting built-in

### Football-Data CSV
- **Polite delay**: 0.5s between requests
- **No hard limit**: Public CSV files

## Dependencies

Required packages:
```
requests
pandas
python-dotenv
pyyaml
urllib3
```

## Testing

Run examples:
```bash
python examples/data_extraction_example.py
```

## Notes

1. **API Key Required**: For API-Football, add your RapidAPI key to `.env`
2. **CSV Always Works**: CSV extraction works without API keys
3. **Versioning**: All data is timestamped and versioned
4. **Metadata**: Each extraction includes metadata for reproducibility
5. **Logging**: Check `logs/` directory for detailed logs

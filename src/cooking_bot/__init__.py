from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys
REPO_PATH = Path(__file__).parent.parent.parent

assert load_dotenv(REPO_PATH/ ".env"), f'Cant load dot env at {REPO_PATH/ ".env"}'

from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler with desired formatting
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
)

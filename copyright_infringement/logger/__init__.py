import logging
import os
from from_root import from_root
from datetime import datetime

# Define log file name and directory
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = 'logs'

# Ensure the log directory path is created based on the root directory
logs_path = os.path.join(from_root(), log_dir, LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set to INFO or higher to avoid debug-level clutter
)

# Suppress DEBUG logs from pymongo and other noisy libraries
logging.getLogger("pymongo").setLevel(logging.WARNING)
# Add additional libraries if necessary, e.g., `logging.getLogger("other_library").setLevel(logging.WARNING)`

# Log initialization confirmation
logging.info("Logging setup complete.")

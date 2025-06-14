import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import logging
from pathlib import Path
from config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingFileHandler(FileSystemEventHandler):
    """Handle streaming file changes"""
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
            logger.info(f"New file detected: {file_path}")
            self.process_new_file(file_path)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
            logger.info(f"File modified: {file_path}")
            self.process_new_file(file_path)
    
    def process_new_file(self, file_path: Path):
        """Process new or modified file"""
        try:
            # Run ETL
            logger.info("Running ETL...")
            subprocess.run(['python', 'etl.py'], check=True)
            
            # Run forecast
            logger.info("Running forecast...")
            subprocess.run(['python', 'run_forecast.py'], check=True)
            
            logger.info("Processing completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing file: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

def main():
    """Main function"""
    # Create data directory if it doesn't exist
    data_dir = Path(RAW_DATA_DIR)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created directory {RAW_DATA_DIR}")
    
    # Set up Observer
    event_handler = StreamingFileHandler()
    observer = Observer()
    observer.schedule(event_handler, RAW_DATA_DIR, recursive=False)
    
    try:
        logger.info(f"Starting to watch directory {RAW_DATA_DIR}...")
        observer.start()
        
        # Run initial ETL
        logger.info("Running initial ETL...")
        subprocess.run(['python', 'etl.py'], check=True)
        
        # Run initial forecast
        logger.info("Running initial forecast...")
        subprocess.run(['python', 'run_forecast.py'], check=True)
        
        # Keep script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping watch...")
        observer.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        observer.stop()
    
    observer.join()

if __name__ == '__main__':
    main() 
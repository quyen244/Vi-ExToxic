# --- CẤU HÌNH LOGGING ---
import logging 

# logging information 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("annotation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)



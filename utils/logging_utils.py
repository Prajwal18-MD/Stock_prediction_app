import logging

def setup_logging():
    logging.basicConfig(filename='app.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message):
    logging.info(message)

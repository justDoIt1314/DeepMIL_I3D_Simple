import logging
import os
from logging.handlers import RotatingFileHandler
def preduceLog(logPath = 'Logs'):
    os.makedirs(logPath,exist_ok=True)
    logger = logging.getLogger('Robot')
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    file_handler = RotatingFileHandler(os.path.join(logPath, 'info.log'),
    maxBytes=50 * 1024 * 1024,
    backupCount=10,
    encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
    
if __name__ == '__main__':
    logger = preduceLog()
    logger.info('sfds')
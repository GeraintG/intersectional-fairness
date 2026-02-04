import logging
import os

def setup_logger(output_dir, log_file_name="experiment.log"):
    """
    Set up a logger that writes to both console and a file.
    
    Args:
        output_dir: Directory to save the log file
        log_file_name: Name of the log file
        
    Returns:
        Logger object
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger("intersectional_fairness")
    logger.setLevel(logging.INFO)
    
    # Create a file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, log_file_name))
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
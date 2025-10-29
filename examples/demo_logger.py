# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Demo for using parrots logger
"""
import sys

sys.path.append('..')
from parrots.log import set_log_level, logger
from parrots import TextToSpeech

if __name__ == "__main__":
    # Example 1: Use default log level (INFO)
    logger.info("This is an info message with default log level")
    logger.debug("This debug message will NOT be shown (default level is INFO)")
    
    # Example 2: Change log level to DEBUG to see more details
    print("\n" + "="*50)
    print("Setting log level to DEBUG...")
    print("="*50 + "\n")
    set_log_level("DEBUG")
    
    logger.debug("Now debug messages are visible!")
    logger.info("Info messages are still visible")
    
    # Example 3: Use with TextToSpeech
    print("\n" + "="*50)
    print("Testing with TextToSpeech...")
    print("="*50 + "\n")
    
    m = TextToSpeech(
        speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
        speaker_name="MaiMai",
        device="cpu",
    )
    
    # Example 4: Change to WARNING level to reduce output
    print("\n" + "="*50)
    print("Setting log level to WARNING (less verbose)...")
    print("="*50 + "\n")
    set_log_level("WARNING")
    
    logger.debug("This debug message will NOT be shown")
    logger.info("This info message will NOT be shown")
    logger.warning("This warning message WILL be shown")
    logger.error("This error message WILL be shown")
    
    # Example 5: You can also set log level via environment variable
    # export PARROTS_LOG_LEVEL=DEBUG  # in bash
    # set PARROTS_LOG_LEVEL=DEBUG     # in Windows cmd
    print("\n" + "="*50)
    print("Tip: You can also set log level via environment variable:")
    print("  export PARROTS_LOG_LEVEL=DEBUG  # Linux/Mac")
    print("  set PARROTS_LOG_LEVEL=DEBUG     # Windows")
    print("="*50)

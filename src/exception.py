import sys
import logging

import sys
import traceback

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: {file_name} at line {line_number} — Error: {str(error)}"
    else:
        return f"Error: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return str(self.error_message)

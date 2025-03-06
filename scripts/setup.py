from pathlib import Path
import sys
import importlib.util
import os
import re
from datetime import datetime

# ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
PINK = "\033[35m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
RESET = "\033[0m"  # Resets to default color

"""
STEP 1. 
Find info module containing variable names etc. pertaining to this project.
We assume the info module has the same parent directory as this file.
We assume it is the only such file whose name is of the form *info*.py.
"""

# Get the directory of the current script
current_script_dir = Path(__file__).parent

# Find the info.py file in the same directory
info_py_files = list(current_script_dir.glob('*info*.py'))

# Check if exactly one info.py file is found
if len(info_py_files) != 1:
    raise FileNotFoundError("Could not uniquely identify the info.py file")

# Get the name of the info.py file
info_py_file = info_py_files[0]

# Get the module name by removing the .py extension
info_module_name = info_py_file.stem

# Add the current script directory to the system path for imports
sys.path.append(str(current_script_dir))

# Import the info.py module dynamically
info_module = importlib.import_module(info_module_name)

"""
STEP 2.
Access variables needed for the setup.
"""

# Now we can access variables and functions from the info.py module
PROJECT_ENVIRONMENT_VARIABLE = getattr(info_module, 'PROJECT_ENVIRONMENT_VARIABLE', None)
SCRIPTS_DIRECTORY_NAME = getattr(info_module, 'SCRIPTS_DIRECTORY_NAME', None)

"""
STEP 3.
Set project path. 
Assumes an environment variable with name PROJECT_ENVIRONMENT_VARIABLE has been created if working on local machine. 
"""

# Get the project path from the environment variable
PROJECT_ENVIRONMENT_VALUE = os.environ.get(PROJECT_ENVIRONMENT_VARIABLE)

if PROJECT_ENVIRONMENT_VALUE is None:
    raise EnvironmentError(f"Environment variable {PROJECT_ENVIRONMENT_VARIABLE} is not set.")

PROJECT_PATH = Path(PROJECT_ENVIRONMENT_VALUE)

"""
STEP 4.
Set scripts path and add any sub-directories to sys.path as well.
"""
# Define the scripts path using the project path and the name of the scripts directory
SCRIPTS_PATH = PROJECT_PATH / SCRIPTS_DIRECTORY_NAME

# Add SCRIPTS_PATH to sys.path 
sys.path.append(str(SCRIPTS_PATH))

# Iterate through all sub-directories of SCRIPTS_PATH and add them to sys.path
for subdir in SCRIPTS_PATH.rglob('*'):
    if subdir.is_dir():
        sys.path.append(str(subdir))

"""
STEP 5.
Find utilities module for this project.
We assume the utilities module lies directly under SCRIPTS_PATH.
We assume it is the only such file whose name contains utls.py as a substring.
"""

def matches_pattern_in_order(filename, pattern_list):
    # Join common utility patterns into a single regex pattern
    regex_pattern = '|'.join([re.escape(pattern) for pattern in pattern_list])
    regex = re.compile(f'.*({regex_pattern}).*', re.IGNORECASE)
    return bool(regex.match(filename))

# Define a list of common utility file name patterns
common_utility_patterns = ['utl', 'utls', 'util', 'utils', 'utility', 'utilities']

# Find all files in the immediate SCRIPTS_PATH directory
all_files = SCRIPTS_PATH.glob('*')

# Filter the files based on the updated matches_pattern_in_order function
utils_py_files = [
    file for file in all_files if file.is_file() and matches_pattern_in_order(file.name, common_utility_patterns)
]

# Check if exactly one 'utls.py' file is found
if len(utils_py_files) != 1:
    raise FileNotFoundError("Could not uniquely identify the utls.py file")

# Get the path of the 'utls.py' file
utils_py_file = utils_py_files[0]

# Get the module name by removing the .py extension and converting to valid module name
utils_module_name = utils_py_file.stem.replace('-', '_').replace('.', '_')

# utils_module= f'{SCRIPTS_PATH.name}.{utils_module_name}'

# Import the utils.py module dynamically
utils_module = importlib.import_module(utils_module_name)

"""
STEP 6.
Import certain functions from utils_module and use them to create a directory tree and path dictionary.
"""

# Import some functions from our utils_module:
# from utils_module import print_header, get_directory_tree, get_subdirectories

print_header = utils_module.print_header 
get_directory_tree = utils_module.get_directory_tree 
get_subdirectories = utils_module.get_subdirectories 

# Create and display our project directory tree
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print_header(f"Project directory tree as at {current_time}")
print_header(f"Base path: {PROJECT_PATH}\n", level=2)
_, project_directory_tree_string = get_directory_tree(base_path=PROJECT_PATH,
                                                      base_name='project')

# Create a dictionary that maps names to paths for immediate subdirectories of the project directory
print_header("Paths to first-level subdirectories stored in 'path' dictionary\n", level=2)
path = get_subdirectories(base_path=PROJECT_PATH,
                          depth=0,)
for dir_name, dir_path in path.items():
    print(f"├─ path[\'{BLUE}{dir_name}{RESET}\'] = {BLUE}{dir_path}{RESET}")

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import re

# ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
PINK = "\033[35m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
RESET = "\033[0m"  # Resets to default color

try:
    from IPython import get_ipython

    if get_ipython():
        from IPython.display import HTML
        import ipywidgets as widgets
    else:
        widgets = None


        def HTML(*args, **kwargs):
            print("HTML display is not available.")
except ImportError:
    widgets = None


    def HTML(*args, **kwargs):
        print("HTML display is not available.")


def set_widgets(enable: bool = True, disable: bool = False) -> None:
    global widgets
    global HTML
    if enable:
        try:
            from IPython import get_ipython
            if get_ipython():
                from IPython.display import HTML
                import ipywidgets as widgets
            else:
                widgets = None

                def HTML(*args, **kwargs):
                    print("HTML display is not available.")
        except ImportError:
            widgets = None

            def HTML(*args, **kwargs):
                print("HTML display is not available.")
    if disable:
        widgets = None

        def HTML(*args, **kwargs):
            print("HTML display is not available.")


def get_subdirectories(base_path: Path,
                       depth: int = 0,
                       ignore: List[str] = None) -> dict:
    """
    Creates a dictionary with keys as the immediate subdirectories of a specified directory
    and values as Path objects pointing to those directories, ignoring specified subdirectories,
    and limiting to a specified depth.

    Args:
        base_path (Path): The path to the base directory.
        depth (int, optional): The depth of subdirectories to include. Defaults to 0.
        ignore (List[str], optional): A list of subdirectory names to ignore. Defaults to None.

    Returns:
        dict: A dictionary with subdirectory names as keys and Path objects as values.
    """
    if not base_path.is_dir():
        raise ValueError(f"The path {base_path} is not a directory.")

    if ignore is None:
        ignore = ['__pycache__', '.ipynb_checkpoints']

    def get_subdirs_at_depth(current_path: Path, current_depth: int) -> dict:
        if current_depth == depth:
            return {
                subdir.name: subdir
                for subdir in current_path.iterdir()
                if subdir.is_dir() and subdir.name[0] not in {'.', '_'} and subdir.name not in ignore
            }

        subdirs = {}
        for subdir in current_path.iterdir():
            if subdir.is_dir() and subdir.name[0] not in {'.', '_'} and subdir.name not in ignore:
                deeper_subdirs = get_subdirs_at_depth(subdir, current_depth + 1)
                subdirs.update(deeper_subdirs)
        return subdirs

    return get_subdirs_at_depth(base_path, 0)

def extend_ignore_list(base_path: Path, regex_pattern: str, use_gitignore: bool) -> List[str]:
    """
    Extends the ignore list based on a regex pattern and an optional .gitignore file.

    Args:
        base_path (Path): The base path to search for the .gitignore file.
        regex_pattern (str): The regex pattern to ignore files and folders.
        use_gitignore (bool): If True, parse the .gitignore file for additional paths to ignore.

    Returns:
        List[str]: An extended list of paths to ignore.
    """
    ignore_list = []

    # Add paths that match the regex pattern
    regex = re.compile(regex_pattern)
    for item in base_path.rglob("*"):
        if regex.match(item.name):
            ignore_list.append(item.name)

    # Optionally add paths from .gitignore
    if use_gitignore:
        gitignore_path = base_path / '.gitignore'
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as gitignore_file:
                for line in gitignore_file:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        ignore_list.append(line)

    return ignore_list


def get_directory_tree(base_path: Path,
                       base_name: str,
                       print_paths: bool = True,
                       ignore: List[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    Prints a directory tree and creates dictionaries mapping directory names and files to path objects.

    Args:
        base_path (Path): The base path to start from.
        base_name (str): The name to use for the base path in the returned dictionary.
        print_paths (bool): If True, print the paths.
        ignore (List[str]): List of directory names or file names to ignore.

    Returns:
        Tuple[Dict[str, Any], str]: A tuple containing the nested dictionary structure representing the directory tree and the string representation of the tree.
    """
    if ignore is None:
        ignore = ['__pycache__', '.ipynb_checkpoints', '00-template.ipynb']
    also_ignore = extend_ignore_list(base_path=base_path, regex_pattern=None, use_gitignore=True)
    ignore = list(set(ignore).union(set(also_ignore)))

    def create_path_dict(path: Path) -> Dict[str, Any]:
        """
        Recursively creates a nested dictionary of paths including files and directories.
        """
        path_dict = {}

        # Separate and sort directories and files
        directories = sorted([item for item in path.iterdir() if
                              item.is_dir() and item.name not in ignore and not item.name.startswith('.')])
        files = sorted([item for item in path.iterdir() if item.is_file() and item.name not in ignore])

        # Add directories first, then files
        for directory in directories:
            subdirectory = create_path_dict(directory)
            path_dict[directory.name] = {"path": directory, "subdirectories": subdirectory}

        for file in files:
            path_dict[file.name] = {"path": file}

        return path_dict

    # Start with the base path
    path_structure = {base_name: {"path": base_path, "subdirectories": create_path_dict(base_path)}}

    def print_path_dict(d: Dict[str, Any], prefix: str = "", output: str = "") -> str:
        """
        Recursively prints the nested dictionary of paths in a tree format and accumulates the output in a string.
        Directories are printed in blue, and files are printed in yellow.
        """
        for key, value in d.items():
            if "subdirectories" in value:
                # Directory in blue
                output += prefix + f"├─ {BLUE}{key}/{RESET}\n"
                output = print_path_dict(value["subdirectories"], prefix + "│  ", output)
            else:
                # File in yellow
                output += prefix + f"└─ {YELLOW}{key}{RESET}\n"
        return output

    tree_output = print_path_dict(path_structure[base_name]['subdirectories'])

    if print_paths:
        print(tree_output)

    return path_structure, tree_output

def extend_ignore_list(base_path: Path, regex_pattern: str = None, use_gitignore: bool = True) -> List[str]:
    """
    Extends the ignore list based on an optional regex pattern and an optional .gitignore file.

    Args:
        base_path (Path): The base path to search for the .gitignore file.
        regex_pattern (str, optional): The regex pattern to ignore files and folders. Defaults to None.
        use_gitignore (bool): If True, parse the .gitignore file for additional paths to ignore.

    Returns:
        List[str]: A list of normalized ignore patterns.
    """
    ignore_list = []

    # Add paths that match the regex pattern, if provided
    if regex_pattern:
        regex = re.compile(regex_pattern)
        for item in base_path.rglob("*"):
            if regex.match(item.name):
                ignore_list.append(item.name)

    # Optionally add paths from .gitignore
    if use_gitignore:
        gitignore_path = base_path / '.gitignore'
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as gitignore_file:
                for line in gitignore_file:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Remove leading '/' or '**/' and trailing '/' if present
                        line = line.lstrip("/").lstrip("**/").rstrip("/")
                        ignore_list.append(line)

    return ignore_list


# USEFUL FUNCTIONS

try:
    from IPython.display import display as ipy_display
except ImportError:
    ipy_display = None


def display(*args):
    if ipy_display is not None:
        for arg in args:
            if arg is not None:
                ipy_display(arg)
    else:
        for arg in args:
            if arg is not None:
                print(arg)

try:
    from IPython.display import clear_output as ipy_clear_output
except ImportError:
    ipy_clear_output = None
try:
    from IPython.display import clear_output as ipy_clear_output
except ImportError:
    ipy_clear_output = None


def clear_output():
    if ipy_clear_output is not None:
        ipy_clear_output(wait=True)
    else:
        # Fallback to clearing the terminal if not in a Jupyter environment
        import os
        if os.name == "nt":  # Windows
            os.system("cls")
        else:  # Unix-based systems (Linux, macOS)
            os.system("clear")


def get_variable_name(var):
    # Get the name of the variable using the inspect module
    for name, value in inspect.currentframe().f_back.f_locals.items():
        if value is var:
            return name
    return None  # Variable name not found


def save_df(df: pd.DataFrame,
            save_as: Union[str, Path],
            tack_on: Union[str, None] = None,
            index: bool=False,) -> None:
    # If filename is a Path object, convert it to a string
    if isinstance(save_as, str):
        save_as = Path(save_as)

    # Insert suffix before the file extension
    if tack_on is not None:
        save_as = save_as.with_name(f"{save_as.stem}_{tack_on}{save_as.suffix}")

    # Determine the file extension
    file_extension = save_as.suffix.lower()

    # Choose the saving method based on the file extension
    if file_extension == '.csv':
        df.to_csv(save_as, index=index)
    elif file_extension == '.xlsx':
        df.to_excel(save_as, index=index)
    elif file_extension == '.html':
        df.to_html(save_as, index=index)
    elif file_extension == '.json':
        df.to_json(save_as, orient='records')
    elif file_extension == '.png':
        plt.axis('off')  # Turn off the axis
        plt.table(cellText=df.values, colLabels=df.columns, loc='center')  # Display DataFrame as a table
        plt.savefig(save_as)  # Save as PNG
    elif file_extension == '.txt':
        # Convert DataFrame to dictionary
        df_dict = df.to_dict()
        # Save dictionary as text file
        with open(save_as, 'w') as file:
            file.write(str(df_dict))
    else:
        print(f"Unsupported file extension: {file_extension}. Please use csv, xlsx, html, or json.")


def print_header(header: str, level: Union[None,int]=None,) -> None:
    if level is None or level == 1:
        length = len(header)
        output_sequence = ["", "=" * length, header.upper(), "=" * length, ""]
        output = '\n'.join(output_sequence)
    else:
        output = f"{header.upper()}"
    print(f"{PINK}{output}{RESET}")
    return None

def display_aligned(*strings: str) -> None:
    # Split each string at '=', storing the left and right parts
    parts = [s.split('=', 1) for s in strings if '=' in s]

    # Find the maximum length of the left part for alignment
    max_left_length = max(len(part[0].strip()) for part in parts)

    # Display each string with alignment on the '=' symbol
    for left, right in parts:
        print(f"{left.strip():>{max_left_length}} = {right.strip()}")

def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0  # Initialize call count
    return wrapper
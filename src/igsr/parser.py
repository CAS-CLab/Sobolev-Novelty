import ast
import re
from typing import Any, List, Optional, Tuple, Union


def extract_python_list(text: str) -> Optional[List[Any]]:
    """
    Extract a Python parsable list of literals from a string, supporting multi-line lists.

    Args:
        text (str): The input string that may contain a Python list.

    Returns:
        Optional[List[Any]]: The extracted list if found, None otherwise.
    """
    # Regular expression to find potential Python lists in the text.
    # Using re.DOTALL (s) flag to make the dot match newlines as well.
    list_pattern = r"\[.*?\]"

    # Find all potential list matches with DOTALL flag.
    potential_lists = re.findall(list_pattern, text, re.DOTALL)

    for potential_list in potential_lists:
        try:
            # Try to parse the potential list using ast.literal_eval.
            parsed_list = ast.literal_eval(potential_list)

            # Check if the result is actually a list.
            if isinstance(parsed_list, list):
                return parsed_list
        except (SyntaxError, ValueError):
            # If parsing fails, this wasn't a valid Python list.
            continue

    # Return None if no valid list was found.
    return None


def extract_code_blocks(
    text: str,
    *,
    return_info_string: bool = False,
    sentinel: Optional[str] = None,
) -> Union[List[Tuple[Optional[str], str]], List[str]]:
    """
    Extract triple-backtick code blocks from `text`.

    Note:
        - Supports blocks with or without an info string (e.g. <3 backticks>python ...)
        - Supports multi-line and single-line blocks (e.g. <3 backticks>print("hi")<3 backticks>)
        - Handles CRLF line endings - converts them to LF so the body and the info-string never contain stray '\r' characters.
        - If `sentinel` is given, **only** blocks whose opening fence appears immediately after the sentinel text are extracted.

      sentinel + '```'   ←  must be contiguous
      (sentinel may contain new-lines; it must **not** contain '`')

    Args:
        text (str): The input text containing code blocks.

    Returns:
        If `return_info_string` is True:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains:
                - str: The language identifier (e.g. "python"), a.k.a. info-string.
                - str: The code block content.
        Otherwise:
            List[str]: A list of code block contents.
    """
    if sentinel is not None:
        if "`" in sentinel:
            raise ValueError("sentinel may not contain back-tick characters")
        sentinel_needle = sentinel + "```"

    newline_fence_re = re.compile(r"(?:\r?\n)[ \t]*```[ \t]*(?:\r?\n|$)")
    fence_len = 3
    blocks: List[Tuple[Optional[str], str]] = []
    pos = 0

    while True:
        # Locate an opening fence (plain or prefixed by sentinel):
        if sentinel is None:
            open_idx = text.find("```", pos)
            if open_idx == -1:
                break
        else:
            hit = text.find(sentinel_needle, pos)
            if hit == -1:
                break
            open_idx = hit + len(sentinel)  # Index of the first `

        content_start = open_idx + fence_len
        close_idx = text.find("```", content_start)
        if close_idx == -1:
            break  # Unterminated - stop.

        first_nl = text.find("\n", content_start, close_idx)
        is_inline = first_nl == -1

        if is_inline:
            body = text[content_start:close_idx].replace("\r\n", "\n").rstrip()
            blocks.append((None, body))
            pos = close_idx + fence_len
            continue

        # Multi-line: pick info-string (may be empty) and find a *proper* close:
        info = text[content_start:first_nl].strip() or None
        close_match = newline_fence_re.search(text, first_nl)
        if not close_match:
            break  # No closing fence.

        body = text[first_nl + 1 : close_match.start()]
        body = body.replace("\r\n", "\n").rstrip("\n")
        blocks.append((info, body))
        pos = close_match.end()

    return blocks if return_info_string else [body for _, body in blocks]

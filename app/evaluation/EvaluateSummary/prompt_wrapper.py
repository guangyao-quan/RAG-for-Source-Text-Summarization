"""
This module provides a mechanism for applying a predefined prompt template to an original document
before passing it to a function. The primary use case is to format the document for summarization
tasks, ensuring that the summary adheres to specified guidelines.

Classes and Functions:
----------------------

1. `PromptTemplate`:
    - Represents a template for creating prompts.

2. `with_prompt_template`:
    - A decorator that formats the original document using a predefined prompt template before
      passing it to the decorated function.

Variables:
----------

1. `template` (str):
    - A string containing the summarization prompt template, including instructions, requirements,
      and an example.

2. `qa_template` (PromptTemplate):
    - An instance of `PromptTemplate` initialized with the `template`.

Functions:
----------

1. `with_prompt_template(func)`:
    - A decorator function that wraps another function, formatting its first argument (the original
      document) using the prompt template before calling the function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function that formats the input document with the prompt template.

Usage Example:
--------------

    @with_prompt_template
    def summarize(self, prompt):
        # Function implementation
        pass

    # The function `summarize` will now receive the formatted prompt instead of the original document.

"""

from functools import wraps
from llama_index.core import PromptTemplate

# Define the template for summarization prompts
template = (
    "## Summarization Prompt Template\n\n"

    "We have provided the original document to be summarized below.\n"
    "---------------------\n"
    "### Original Document:\n"
    "{original_document}\n"
    "---------------------\n\n"

    "### Instructions:\n"
    "Using the above document, create a concise and precise summary. "
    "Focus on the key points and essential information, "
    "ensuring that the summary is as brief as possible while still capturing the main ideas.\n\n"

    "### Requirements:\n"
    "1. **Brevity**: Keep the summary IN ONE (EXTREMELY) SHORT SENTENCE and to the point.\n"
    "2. **Clarity**: Ensure the summary is clear and easy to understand.\n"
    "3. **Coverage**: Include all critical information from the original document.\n"
    "4. **Relevance**: Exclude any extraneous details.\n\n"

    "---------------------\n"
    "### Example Document:\n"
    "Media playback is unsupported on your device 8 May 2015 Last updated at 10:28 BST During the war, "
    "families would have to ration their food and had little communication in their homes. Luxuries "
    "like chocolate and fruit were very difficult to find and families had to grow their own food to "
    "survive. Watch Martin's report to find out more.\n\n"

    "### Example Summary:\n"
    "Martin went to the German Occupation Museum to see what life was like for a family living on Guernsey in "
    "World War II.\n"
    "---------------------"
)
qa_template = PromptTemplate(template)


def with_prompt_template(func):
    """
    Decorator to format the original document using a predefined prompt template before
    passing it to the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function that formats the input document with the prompt template.
    """
    @wraps(func)
    def wrapper(self, *args):
        """
        Wrapper function that applies the prompt template to the input document.

        Args:
            self: The instance of the class (if the decorated function is a method).
            *args: Positional arguments to the decorated function, where the first argument
                   is expected to be the original document.

        Returns:
            The result of the decorated function called with the formatted prompt.
        """
        prompt = qa_template.format(original_document=args[0])
        return func(self, prompt)
    return wrapper

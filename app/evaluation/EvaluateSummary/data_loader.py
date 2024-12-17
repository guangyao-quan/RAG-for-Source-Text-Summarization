"""
This module defines the EvaluationDataGenerator class which is designed to load and generate
evaluation data for machine learning or data processing applications. The class utilizes the
`datasets` library to load data from a specified path and configuration, and offers functionality
to concatenate different splits of the dataset into a single iterable dataset.

Example usage:
    >> generator = EvaluationDataGenerator(path='path/to/dataset', configuration_name='default')
    >> batch = generator.generate(num_samples=100)
    >> if batch is None:
           print("No more data to generate.")
    >> generator.reset_index()  # Reset to begin generating from the start again.
"""

from itertools import islice

from datasets import IterableDataset, concatenate_datasets, load_dataset


class EvaluationDataGenerator:
    """
    A class to load and generate evaluation data for machine learning or data processing applications.

    Attributes:
        path (str): The path to the dataset.
        configuration_name (str or None): The configuration name for the dataset.
        iterator (Iterator): An iterator for the loaded dataset.

    Methods:
        _load_dataset(): Loads and concatenates the dataset splits into a single iterable dataset.
        generate(num_samples): Generates a batch of samples from the dataset.
        reset_index(): Resets the index of the iterator to the beginning.
    """

    def __init__(self, path: str, configuration_name: str | None) -> None:
        """
        Initializes the EvaluationDataGenerator with the specified dataset path and configuration.

        Args:
            path (str): The path to the dataset.
            configuration_name (str or None): The configuration name for the dataset.
        """
        self.path = path
        self.configuration_name = configuration_name
        self.iterator = iter(self._load_dataset())

    def _load_dataset(self) -> IterableDataset:
        """
        Loads the dataset using the specified path and configuration, concatenating all splits into a single iterable dataset.

        Returns:
            IterableDataset: The loaded and concatenated dataset.
        """
        dataset = load_dataset(self.path, self.configuration_name, keep_in_memory=False)
        dataset = concatenate_datasets(
            [dataset[split] for split in dataset.keys()]
        ).to_iterable_dataset()
        return dataset

    def generate(
        self, num_samples: int = 100, max_length: int | None = None
    ) -> list[dict] | None:
        """
        Generates a batch of samples from the dataset, optionally filtering by maximum text length.

        Args:
            num_samples (int): The number of samples to generate. Defaults to 100.
            max_length (int | None): Optional maximum length of the text of the samples. If specified,
                                     only samples with text length less than this value are included.

        Returns:
            list[dict] or None: A list of generated samples or None if no more data is available.
        """
        if max_length is not None:
            # Create a filtered iterator if max_length is specified
            filtered_iterator = (
                item for item in self.iterator if len(item["document"]) < max_length
            )
            batch = list(islice(filtered_iterator, num_samples))
        else:
            # Use the original iterator if no max_length is specified
            batch = list(islice(self.iterator, num_samples))

        return batch if len(batch) > 0 else None

    def reset_index(self) -> None:
        """
        Resets the index of the iterator to the beginning.
        """
        self.iterator = iter(self._load_dataset())

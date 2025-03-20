import argparse
import inspect


class Parser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        valid_args = inspect.signature(argparse.ArgumentParser).parameters

        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_args}

        invalid_args = set(kwargs) - set(filtered_kwargs)
        if invalid_args:
            raise ValueError(f'Error ! Invalid arguments passed on parser instantiation  : {invalid_args}')

        super().__init__(**filtered_kwargs)

    def get_args(self, parser_dict: dict[str, any]):
        """
        Add arguments from parser dictionary

        Args:
            parser_dict: Dictionary containing argument configurations
        """
        for arg_name, properties in parser_dict.items():
            # Get the flags list and convert to tuple if it's not already
            flags = properties['flags']
            if isinstance(flags, list):
                flags = tuple(flags)

            # Create a copy of properties to modify
            kwargs = properties.copy()
            # Remove flags from kwargs as they're passed separately
            kwargs.pop('flags')

            # Add the argument to the parser
            try:
                self.add_argument(*flags, **kwargs)
            except Exception as e:
                print(f"Error adding argument {flags}: {str(e)}")
                raise

        try:
            args = self.parse_args()
            return args
        except Exception as e:
            print(f"Error parsing arguments: {str(e)}")
            raise
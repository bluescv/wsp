class ControlError(Exception):
    """Only a fixed code crosses the CLI boundary; never include input values."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)

class userInputError(Exception):
    """Raises when a user inputs invalid ETF input data."""
    pass

class datasetFormationError(Exception):
    """Raises when an error occurs while creating components of a data set."""
    pass

class faultyDatasetError(Exception):
    """Raises when an error is spotted in the input dataset"""
    pass

class modelConfigurationError(Exception):
    """Raises when invalid model parameters are given"""
    pass
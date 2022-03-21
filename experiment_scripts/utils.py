import os


def check_write_permission(filename: str) -> bool:
    """Returns True if the file is writable, else False.
    Parameters
    ----------
    filename : str
        the file we want to check the writability of
    Returns
    -------
    bool
        writable status of the given filename
    """
    assert isinstance(filename, str), f"invalid filename: {filename}"

    if os.path.isfile(filename):
        writable = os.access(filename, os.W_OK)
    else:
        # we need write and execute access in order to create new files
        writable = os.access(os.path.dirname(filename), os.W_OK | os.X_OK)

    return writable



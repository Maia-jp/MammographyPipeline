import os
def safe_make_folder(i):
    """Makes a folder if not present
    """
    if not os.path.exists(i):
        os.makedirs(i)
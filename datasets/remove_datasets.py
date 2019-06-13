
import  datasets.load_datasets as ld
import shutil

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    Parameters
    ----------
    data_home : str | None
        The path to attngan data dir.
    """
    data_home = ld.get_data_home(data_home)

    logger.info(f"Removing the data directory: {data_home}")
    shutil.rmtree(data_home)

    return

def clear_data_set(data_set, data_home=None):
    pass
    #NotImplemented
    # """Deletes the data set specified in data_home.
    # Parameters
    # ----------
    # data_home : str | None
    #     The path to attngan data dir.
    # """
    # data_home = ld.get_data_home(data_home)

    # logger.info(f"Removing the data directory: {data_home}")
    # shutil.rmtree(data_home)


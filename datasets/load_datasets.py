"""
Loads datasets to train algorithms and generate images for supported text to image algorithms.


"""
from collections import namedtuple
import pandas as pd
import os



RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


def get_data_home(data_home=None):
    """Return the path of the pytig data dir.
    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data dir is set to a folder named 'data' in the
    user home folder.
    # Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    # variable or programmatically by giving an explicit folder path. The '~'
    # symbol is expanded to the user home folder.
    # If the folder does not already exist, it is automatically created.
    # Parameters
    ----------
    data_home : str | None
        The path to scikit-learn data dir.
    """
    if data_home is None:
        data_home = os.environ.get('DATA',
                                os.path.join('~', 'AttnGAN/data')) #TODO: Generalize to pytig
    data_home = os.path.expanduser(data_home)

    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home

def photosynthesis_raw(data_home=None):

    # load text data from pytig-data
    #txt_files =


    #
    data_home = get_data_home(data_home)


    pass

def photosynthesis_lbld():
    pass
''' Purpose: A set of functions to load data into the AttnGAN model for training
    Notes:
        - Very path dependant
'''

from google_images_download import google_images_download
import os



def download_images(term ,img_args):
    
    # Download Images 
    response = google_images_download.googleimagesdownload()
    img_paths = response.download(img_args)
    
    return img_paths, response



def handle_missing_directories(directory_flpth):
    # Handle missing Directory
    if not os.path.exists(directory_flpth):
        
        os.makedirs(directory_flpth)
        print("Made new directory: {}".format(directory_flpth))
        # print(os.path.join(dirname, flpth))
    else:
        pass
    
    return
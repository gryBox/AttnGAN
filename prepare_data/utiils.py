


import os


def handle_missing_directories(directory_flpth):
    # Handle missing Directory
    if not os.path.exists(directory_flpth):
        
        os.makedirs(directory_flpth)
        print("Made new directory: {}".format(directory_flpth))
        # print(os.path.join(dirname, flpth))
    else:
        pass
    
    return
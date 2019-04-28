''' Prepare Data files for training models invoked when a model needs to be trained from scratch or new
    train and validation sets need to be remeade
'''


import miscc.utils as utils

import pathlib
import pickle
import glob
import numpy as np

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import os



class PrepareDataset_for_ModelTraining(object):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        #logging.critical(self.cfg)
        
        
        # Initialize all the filepaths
        self.path_to_captions_pickle = os.path.join(self.cfg.DATA_DIR, 'captions.pickle')
        self.filenamesText_flpth = os.path.join(self.cfg.DATA_DIR, 'filenames.txt')
        self.train_flpth = os.path.join(self.cfg.DATA_DIR, 'train')
        
        
        # Get all the names of files from the text directory
        self.captionsFilename_lst = glob.glob(cfg.TEXT_DIR + '/**/*.txt', recursive=True)
        logging.info("Number of text files: {}".format(len(self.captionsFilename_lst)))
        #logging.critical("Text files: {}".format(self.captionsFilename_lst))
        
        ## Transform the filenames to a relative path to the text directory 
        self.modifiedCaptionsFilename_lst = [self.modified_txt_flpth(relCaption_flpth)  for relCaption_flpth in self.captionsFilename_lst]
        #logging.critical("Modified text files: {}".format(self.modifiedCaptionsFilename_lst))
        ######################
        # 1. Remove caprions pickle file
        utils.remove_file( self.path_to_captions_pickle)
        
        # 2. Write text filenames to a filenames file
        self.writeFilenames_to_txt(self.filenamesText_flpth, self.modifiedCaptionsFilename_lst)
        
        # 3. Split Files between training and test data according to config settings
        self.split_dict = self.splitData(self.cfg.TRAIN_SPLIT, self.cfg.VALIDATION_SPLIT, self.modifiedCaptionsFilename_lst)
        
        # 4. Pickle split textfilenames list andwrite tp directories for models
        self.processedData_filenames = self.pickleFilenames_lst(self.split_dict) 
        
        logging.info("FINISHED WRITING TEXT TO <<{}>>".format(self.processedData_filenames))
    
    def pickleFilenames_lst(self, split_dict):
        pickled_filename_dict = dict()
        for splitType in self.split_dict:
            
            logging.debug('splitType: {} '.format(splitType))
            lst_to_write = self.split_dict[splitType]
            
            txtFlpth_pickle = os.path.join(self.cfg.DATA_DIR , splitType, 'filenames.pickle')
            self.txtFilenamesTo_pickle(txtFlpth_pickle, lst_to_write)
            logging.debug("Number of items in pickled list: {}".format(len(lst_to_write)))
            
            pickled_filename_dict[splitType] = txtFlpth_pickle
            
            
        return pickled_filename_dict
    
    def writeFilenames_to_txt(self, filenamesText_flpth, modifiedCaptionsFilename_lst):
        try:
            
            print(filenamesText_flpth)
            with open(filenamesText_flpth, 'w') as f:
                # Writes
                f.write("\n".join(str(filename) for filename in modifiedCaptionsFilename_lst)) 
                
        except Exception as err:
            logging.debug("File not written - Error(s): {}".format(err))
        
        return
    
    # Makes a relative path to the text files
    def modified_txt_flpth(self, full_flpth):
        # Read in filepath and seperate
        p = pathlib.Path(full_flpth)
        txtRel_flpth = p.relative_to(self.cfg.DATA_DIR)
        txtCaption_flpth = str(txtRel_flpth.parent)
        
        txtFile_nm = txtRel_flpth.stem
        #print("Filename: {}".format(txtFile_nm))    
        
        
        caption_flpth = os.path.join(txtCaption_flpth, txtFile_nm)
        
        return  p.stem # caption_flpth
    
    # Seperate the filenames to train data and test\cross validation data
    def splitData(self, trainSplit, testSplit, filename_lst):
        
        # Calculate total number of filenames
        num_filenames = len(filename_lst)
    
        numTrain_files = np.ceil(trainSplit * num_filenames).astype(int)
        numTest_files = np.floor(testSplit * num_filenames).astype(int)
        
        print("Number of Train files: {}".format(numTrain_files))
        print("Number of Test files: {}".format(numTest_files))
        
        trainFile_lst = filename_lst[:numTrain_files]
        testFile_lst = filename_lst[-numTest_files:]
        
        return {
            "train": trainFile_lst,
            "test": testFile_lst
        }
    
    def txtFilenamesTo_pickle(self, flpth, lst_to_write):
        
        utils.handle_missing_directories(os.path.dirname(flpth))
        
        utils.remove_file(flpth)
        
        pickle_out  = open(flpth, "wb")
        pickle.dump(lst_to_write, pickle_out)
        pickle_out.close()
    
        return 


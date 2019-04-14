

import textacy
import en_core_web_sm

import os






# Create text file for each doc - Each Doc maps to an image
def convertCorpus_to_imageLabels(corpus, term, flpth='../data'):
    
    # Loop through corpus and write document to flpth (s3)
    ''' Each doc in a corpus equals and image'''
    for ix, doc in enumerate(corpus):
        print(doc.n_sents)
        
        # Write to file stuff (paths)
        filename = "{}_{}.txt".format(term, ix)
        path_to_file = "{}/{}".format(flpth, filename)
        
        # Handle missing Directory
        if not os.path.exists(flpth):
            dirname = os.path.abspath('')
            os.makedirs( os.path.join(dirname, flpth))
            print(os.path.join(dirname, flpth))
        
        
        f =  open(path_to_file, 'w')
        
        # Prepeare labals for an image
        for sent in doc.sents:
            label = textacy.preprocess.preprocess_text(sent.text,
                                               lowercase=True,
                                               no_punct=True
                                              )
            f.write(label+"\n" )
            
        f.close()
    return ix
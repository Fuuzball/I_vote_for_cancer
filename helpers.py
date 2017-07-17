# helper functions for dealing with text

import pandas as pd
import numpy as np

def get_unique_text(variants_df, text_df, cls, save = None, suppress_output = True):
    """
    Takes pandas DataFrames for variants and text files. The
    class number should be an int corresponding to the tumor
    class. Returns a string with all of the unique text
    corresponding to cls.

    suppress_output suppresses printing additional information
    """

    matching_ids_df = variants_df.loc[variants_df['Class'] == cls]['ID'] # all IDs matching cls
    class_text_df = text_df.loc[text_df['ID'].isin(matching_ids_df)]
    # class_text_df is all the text pertaining to cls (including duplicates)
    unique_text = class_text_df['Text'].unique()
    unique_text_string = '\n'.join(unique_text)

    if not suppress_output:
        cls_size = class_text_df.size
        unique_size = unique_text.shape[0]
        print('Number of entries in class {}: {}'.format(cls, cls_size))
        print('Number of unique texts: {}'.format(unique_size))

    if save is not None:
        print('Saving to file: {}'.format(save))
        f = open(save, 'wt')
        f.write(unique_text_string)
        f.close()

    return unique_text_string
    

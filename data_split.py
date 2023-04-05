import os
import os.path as osp
import pandas as pd
from Constants import CATEGORY_INDEX
from fast_ml.model_development import train_valid_test_split

def split_data(directory, train_split=80, test_split=10, validation_split=10):

    train_data = []
    val_data = []
    test_data = []

    ds_store = '.DS_Store'
    
    action_list = os.listdir(directory)
    if ds_store in action_list: action_list.remove(ds_store)

    for action_name in action_list:

        video_list = os.listdir(osp.join(directory,action_name))
        if ds_store in video_list: video_list.remove(ds_store)

        video_df = pd.DataFrame(video_list,columns=['video_idx'])
        video_df['action_idx'] = CATEGORY_INDEX[action_name]
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(video_df,target='action_idx',train_size=train_split/100, valid_size=validation_split/100, test_size=test_split/100)
        train_data.extend(X_train['video_idx'])
        val_data.extend(X_valid['video_idx'])
        test_data.extend(X_test['video_idx'])
    
    return train_data,val_data,test_data
            
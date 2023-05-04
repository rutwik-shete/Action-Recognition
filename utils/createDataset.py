import os
import os.path as osp
from Constants import CATEGORY_INDEX
import shutil

def createDataset(sourceDirectory,block_size,home_dir):
    destDirectory = osp.join(home_dir,"CustomDataset_{:2d}_frames".format(block_size))

    ds_store = '.DS_Store'

    if not osp.exists(destDirectory):
        print("The dataset with {:2d} frames does not exists : ".format(block_size))
        action_list = os.listdir(sourceDirectory)
        if ds_store in action_list: action_list.remove(ds_store)

        for action_id,action_name in enumerate(action_list): # Goes through classes
            print("Working On",action_name,"... {:2d}/{:2d}".format(action_id+1,len(action_list)))
            video_list = os.listdir(osp.join(sourceDirectory,action_name))
            if ds_store in video_list: video_list.remove(ds_store)

            for video_id in video_list: # Goes through videos
                
                set_id = 0
                block_count = 0

                frame_ids = os.listdir(osp.join(sourceDirectory,action_name,video_id))
                frame_ids = sorted(frame_ids, reverse=False)
                if ds_store in frame_ids: frame_ids.remove(ds_store)

                for video_frame_id in frame_ids: # Goes Through Frames
                    frame_path = osp.join(sourceDirectory,action_name,video_id,video_frame_id)
                    
                    if block_count%block_size == 0:
                        set_id+=1
                    
                    block_count+=1

                    set_name = str(video_id) +"_"+ str(set_id)
                    set_path = osp.join(destDirectory,action_name,video_id,set_name)

                    if not os.path.exists(set_path):
                        os.makedirs(set_path)

                    # if set_name not in os.listdir(set_path):
                    shutil.copy(frame_path, set_path)

                    
                    
                set_name = str(video_id) +"_"+ str(set_id)
                set_path = osp.join(destDirectory,action_name,video_id,set_name)

                count_frames = len(os.listdir(set_path))

                if count_frames < block_size:
                    shutil.rmtree(set_path, ignore_errors=True)
    else:
        print("The dataset with {:2d} frames exists".format(block_size))

    return destDirectory


# createDataset("/mnt/fast/nobackup/users/rs01960/AML/HMDB_simp",8,"/mnt/fast/nobackup/users/rs01960/AML")
import os
import os.path as osp
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from Constants import CATEGORY_INDEX
import torchvision.transforms as T


def read_image(img_path,transform):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise OSError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            got_img = True
        except OSError:
            print(
                f'IOError incurred when reading "{img_path}". Will redo. Don\'t worry. Just chill.'
            )
            pass
    return img

class BlockFrameDataset(Dataset):
    def __init__(self, directory, video_ids_to_include, block_size=8):
        self.instances, self.labels = self.read_dataset(directory,video_ids_to_include,block_size)

        # convert them into tensor
        # self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)
        self.block_size = block_size
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        )

        # normalize
        # self.zero_center()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):

        frame_set_path = self.instances[idx]
        frame_list = os.listdir(frame_set_path)

        current_block = []

        for frame_path in frame_list:
            frame_set_id_path = osp.join(frame_set_path,frame_path)
            frame = read_image(frame_set_id_path,self.transform)
            current_block.append(frame.numpy())
        
        current_block = np.array(current_block)

        output_frame = current_block.reshape((1, self.block_size, 3, 224, 224))

        output_frame = np.array(output_frame, dtype=np.float32)

        output_frame = torch.from_numpy(output_frame)

        return output_frame, self.labels[idx]

    def zero_center(self):
        # self.instances -= float(self.mean)
        pass

    def read_dataset(self, directory, video_ids_to_include, block_size, mean=None):
        
        # set paths according to split
        filepath = directory


        # accumulate the instances and label
        instances = []
        labels = []

        current_block = []
        
        ds_store = '.DS_Store'

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        )

        action_list = os.listdir(directory)
        if ds_store in action_list: action_list.remove(ds_store)

        for action_name in action_list:
            
            video_list = os.listdir(osp.join(directory,action_name))
            if ds_store in video_list: video_list.remove(ds_store)

            for video_id in video_list:

                if video_id in video_ids_to_include:

                    frame_set_ids = os.listdir(osp.join(directory,action_name,video_id))
                    if ds_store in frame_set_ids: frame_set_ids.remove(ds_store)

                    for video_frame_set_id in frame_set_ids:
                        frame_set_id_path = osp.join(directory,action_name,video_id,video_frame_set_id)

                        instances.append(frame_set_id_path)
                        labels.append(CATEGORY_INDEX[action_name])
                        # frame = read_image(frame_path,transform)
                        # current_block.append(frame.numpy())

                        # # 8 consecutive frames
                        # if len(current_block) % block_size == 0:
                        #     current_block = np.array(current_block)

                        #     instances.append(current_block.reshape((1, block_size, 3, 224, 224)))
                        #     labels.append(CATEGORY_INDEX[action_name])

                        #     current_block = []

        # instances = np.array(instances)
        labels = np.array(labels, dtype=np.uint8)

        # self.mean = np.mean(instances)
                
        return instances,labels
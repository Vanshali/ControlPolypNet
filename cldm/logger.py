import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()
                
                



    @rank_zero_only
    def log_local_val(self, save_dir, split, image_filename_pairs, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for image, filename in image_filename_pairs:
        
        
            if isinstance(image, torch.Tensor):
                
                if self.rescale:
                    image = (image + 1.0) / 2.0  # -1,1 -> 0,1
                image = image.squeeze().cpu().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = (image * 255).astype(np.uint8)

                path = os.path.join(root, os.path.basename(os.path.dirname(filename)), os.path.basename(filename))
                if not os.path.exists(os.path.join(root, os.path.basename(os.path.dirname(filename)))):
                    os.mkdir(os.path.join(root, os.path.basename(os.path.dirname(filename))))
                Image.fromarray(image).save(path)


    def log_img_val(self, pl_module, batch, batch_idx, validation_set_length, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

    
            filenames = batch['target_name']  # Assuming 'target_name' is the key for filenames
            
            max_imgs = validation_set_length
            print(validation_set_length)

            with torch.no_grad():
                images = pl_module.log_images(batch, N=validation_set_length, split=split, **self.log_images_kwargs)
           


            desired_image_type = 'samples_cfg_scale_9.00'
           

            # Retrieve images of the desired type and limit their count
            desired_images = images.get(desired_image_type, None)
            #print(desired_images.shape)
            if desired_images is not None:
                N = max(desired_images.shape[0], max_imgs)
                desired_images = desired_images[:N]
                if isinstance(desired_images, torch.Tensor):
                    desired_images = desired_images.detach().cpu()
                    if self.clamp:
                        desired_images = torch.clamp(desired_images, -1., 1.)
                   

            # Create a list of image-filename pairs for the desired type
            image_filename_pairs = [(desired_image, filename) for desired_image, filename in zip(desired_images, filenames)]

            
            self.log_local_val(pl_module.logger.save_dir, split, image_filename_pairs, pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()




    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
            
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        dataloader_idx = 0
        validation_dataloader = trainer.val_dataloaders[dataloader_idx]  # Access the validation dataloader
        validation_set_length = len(validation_dataloader.dataset)  # Get the length of the validation dataset
        if not self.disabled:
            self.log_img_val(pl_module, batch, batch_idx, validation_set_length, split="val")
            
            
            

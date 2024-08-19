# from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_train_load import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset_test_load import MyTestDataset


# Configs
resume_path = './models/control_sd15_ini.ckpt'
#resume_path = './model-epoch=55.ckpt'

batch_size = 32
logger_freq = 300
#learning_rate = 1e-5
#sd_locked = True
only_mid_control = False

learning_rate = 2e-6
sd_locked = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
checkpoint_callback = ModelCheckpoint(dirpath="./", filename='model-{epoch:02d}', save_top_k=-1)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger,checkpoint_callback], default_root_dir='./chkp_folder/', enable_checkpointing = True, max_epochs=55)
#trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], accumulate_grad_batches=4)  # But this will be 4x slower

# Train!
trainer.fit(model, dataloader)

for i in range(1,201):                # Evaluate the model on first 200 images from test json file
    test_dataset = MyTestDataset(i)
    # Create a test dataloader
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=0)
    trainer.validate(model = model, dataloaders = test_loader)


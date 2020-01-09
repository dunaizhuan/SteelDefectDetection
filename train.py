'''
strategy ： train on random cropped img shape [batch, 1, 256, 256]
            and test in img shape [batch, 1, 1600, 800]
test argumentation
cls  -> resnet 34

Unet -34  + SCSE
LOSS BCE

'''
# control block
############################
CROPED = True
ReLoad = False
############################
from model import get_model
import os
import torch
import gc
from utils import *
from loss import *
from Dataset import *
import time
from torch import optim
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#torch.backends.cudnn.deterministic = True


sample_submission_path = './input/sample_submission.csv'
train_df_path = './input/train.csv'
data_folder = "./input/"
test_data_folder = "./input/test_images"

class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 1}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 8
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=1, factor=0.1, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                # mean=(0.485),
                # std=(0.229),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
                cropped=CROPED
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tqdm.tqdm(dataloader)):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            # print(label)
            # print(images.shape)
            #images = images.type(torch.float16)
            #print(images.dtype)
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.net.train()
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.net.eval()
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()


if __name__ == '__main__':
    model = get_model('res34v4')
    x = torch.rand((1, 1, 256, 1600))
    out = model(x)
    print('out')
    print(out.shape)
    if ReLoad:
        ckpt_path = "./model.pth"
        state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        model = model.cuda()
    print('croped in only defect ? %s' %(CROPED))
    print('ReLoad model ? %s' %(ReLoad))
    # PLOT TRAINING
    model_trainer = Trainer(model)
    model_trainer.start()
    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores  # overall dice
    iou_scores = model_trainer.iou_scores


    def plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
        plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
        plt.title(f'{name} plot');
        plt.xlabel('Epoch');
        plt.ylabel(f'{name}');
        plt.legend();
        plt.show()


    plot(losses, "BCE loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")

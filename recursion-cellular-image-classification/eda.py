import os
from data_loader import loader, val_loader
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR,CyclicLR
from model_k import model_resnet_18
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint
import warnings
from config import _get_default_config
from scheduler import ParamScheduler, return_scale_fn
from losses import FocalLoss


model = model_resnet_18
config = _get_default_config()
MODEL_NAME = config.model

path_data = '/mnt/ssd1/datasets/Recursion_class/'
device = 'cuda'


if config.warm_start:
    checkpoint_name = config.checkpoint_name
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint)
    model.to(device)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
path = '/mnt/ssd1/datasets/Recursion_class'
path_data = path
device = 'cuda'
batch_size = 64
warnings.filterwarnings('ignore')


criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0003, momentum=0.9)

metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}


trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    print("Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} "
          .format(engine.state.epoch,
                  metrics['loss'],
                  metrics['accuracy']))


# scale_fn = return_scale_fn()
# lr_scheduler = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001,
#                         mode='exp_range', gamma=1.1, cycle_momentum=False)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
# lr_scheduler = ParamScheduler(optimizer, scale_fn, 500 * len(loader))


@trainer.on(Events.EPOCH_COMPLETED)
def update_lr_scheduler(engine):
    lr_scheduler.step()
    lr = float(optimizer.param_groups[0]['lr'])
    print("Learning rate: {}".format(lr))


check_pointer = ModelCheckpoint('checkpoint_{}'.format(config.checkpoint_folder), MODEL_NAME,
                                n_saved=3, create_dir=True, save_as_state_dict=True,
                                score_function=lambda engine: engine.state.metrics['accuracy'],
                                require_empty=False, score_name="val_accuracy"
                                )

handler = EarlyStopping(patience=25, score_function=lambda engine: engine.state.metrics['accuracy'],
                        trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)
if config.all:
    name_exp = 'general'
else:
    name_exp = config.experiment
val_evaluator.add_event_handler(Events.COMPLETED, check_pointer,
                                {'{}_site_{}'.format(name_exp, config.site): model})


@trainer.on(Events.EPOCH_STARTED)
def turn_on_layers(engine):
    epoch = engine.state.epoch
    if epoch < 2:
        for name, child in model.named_children():
            if name == 'fc' or name == 'classifier':
                pbar.log_message(name + ' is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                pbar.log_message(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
    elif epoch > 1:
        pbar.log_message("Turn on all the layers")
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True


pbar = ProgressBar(bar_format='')
pbar.attach(trainer, output_transform=lambda x: {'loss': x})
trainer.run(loader, max_epochs=500)


# with torch.no_grad():
#     preds = np.empty(0)
#     for x, _ in tqdm_notebook(tloader):
#         x = x.to(device)
#         output = model_resnet_18(x)
#         idx = output.max(dim=-1)[1].cpu().numpy()
#         preds = np.append(preds, idx, axis=0)
#
#
# submission = pd.read_csv(path_data + '/test.csv')
# submission['sirna'] = preds.astype(int)
# submission.to_csv('submission_1.csv', index=False, columns=['id_code', 'sirna'])

import torch

LEARNING_RATE   = 4.5e-4
EPOCH           = 20
BATCH           = 512
FOLDS_NUM       = 5
DEVICE          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_PATH = "RumorSleuth_v3.pth"

LOAD_MODEL = False
SAVE_MODEL = True

def save_checkpoint(model, filename = CHECKPOINT_PATH):
    """ Saving Checkpoint... """
    torch.save(model.state_dict(), filename)

def load_checkpoint(model, filename = CHECKPOINT_PATH):
    """ Loading pretrained weights. """
    model.load_state_dict(torch.load(filename, map_location=DEVICE, weights_only=True), strict=False)
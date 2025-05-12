import torch
import os
import utils.dataset_utils as Dataset_utils
from torch import cuda
from torch.optim import Adam
from torchkge.sampling import  UniformNegativeSampler
from torchkge.models import ComplExModel
from torchkge.utils import MarginLoss, DataLoader
from utils.multiclassNLL_class import MulticlassNLL
from utils.regularizations_class import Regularization
from config.config import *



def full_training(train_kg,params_GS):

    

    model = ComplExModel(params_GS["EMB_DIM"], train_kg.n_ent, train_kg.n_rel)
    criterion = MulticlassNLL()
    regularizer = Regularization(reg_type=params_GS["REG_TYPE"], reg_const=params_GS["REG_CONST"], p=params_GS["P"])

    optimizer = Adam(model.parameters(), lr=params_GS["LR"])
    sampler = UniformNegativeSampler(train_kg)
    dataloader = DataLoader(train_kg, batch_size=BATCH_SIZE)

    #if cuda.is_available():
    #    cuda.empty_cache()
    #    model.cuda()
    #    criterion.cuda()
    
    trigger_times = 0
    best_loss = float('inf')

    for epoch in range(1, N_EPOCHS+1):  
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()  

            h, t, r = batch[0], batch[1], batch[2]  
            n_h, n_t = sampler.corrupt_batch(h, t, r, n_neg=params_GS["ETA"])

            pos, neg = model(h, t, r, n_h, n_t)
            crit_loss = criterion(pos, neg)
            reg_loss = regularizer.forward(model)
            loss = crit_loss + reg_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch} | Mean Loss: {avg_loss:.5f}')

        #manual early stop
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0  
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print("Early stopping: la loss non migliora da {} epoche.".format(PATIENCE))
                break


    model.normalize_parameters()
    model_path = os.path.join(LOAD_DIR_MODEL, MODEL_FILENAME)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in: {model_path}")

    #################################################################################

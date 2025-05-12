import torch
import utils.dataset_utils as Dataset_utils
import os
from torch import cuda
from torch.optim import Adam
from torchkge.sampling import UniformNegativeSampler
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import ComplExModel
from torchkge.utils import  DataLoader
from itertools import product
from utils.multiclassNLL_class import MulticlassNLL
from utils.regularizations_class import Regularization
from utils.load_and_save import save_to_json
from config.config import * 






def grid_search(PARAMS_GRID, B_SIZE,train_kg,val_kg):

    best_mrr = -1
    best_params = None
    best_model_state_dict=None
    B_SIZE=B_SIZE
  
    #cartesia product for all the combinations
    param_combinations = list(product(*PARAMS_GRID.values()))
    param_keys = list(PARAMS_GRID.keys())
    
    sampler = UniformNegativeSampler(train_kg)
    dataloader = DataLoader(train_kg, batch_size=B_SIZE)

    for param_comb in param_combinations:
        param_dict = dict(zip(param_keys, param_comb))

        EMB_DIM = param_dict["EMB_DIM"]
        LR = param_dict["LR"]
        ETA = param_dict["ETA"]
        N_EPOCHS = param_dict["N_EPOCHS"]
        REG_TYPE = param_dict["REG_TYPE"]
        P = param_dict["P"]
        REG_CONST = param_dict["REG_CONST"]
        
        model = ComplExModel(EMB_DIM, train_kg.n_ent, train_kg.n_rel)
        criterion = MulticlassNLL()
        regularizer = Regularization(reg_type=REG_TYPE, reg_const=REG_CONST, p=P)
        optimizer = Adam(model.parameters(), lr=LR)

        #if cuda.is_available():
        #    cuda.empty_cache()
        #    model.cuda()
        #    criterion.cuda()
            
        best_loss = float('inf')
        trigger_times = 0
        for epoch in range(1, N_EPOCHS+1):  
            model.train()
            running_loss = 0.0

            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()  
                h, t, r = batch[0], batch[1], batch[2]  
                n_h, n_t = sampler.corrupt_batch(h, t, r, n_neg=ETA)

                pos, neg = model(h, t, r, n_h, n_t)
                crit_loss = criterion(pos, neg)
                reg_loss = regularizer.forward(model)
                loss = crit_loss + reg_loss
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            avg_loss = running_loss / len(dataloader)
    
            if avg_loss < best_loss:
                best_loss = avg_loss
                trigger_times = 0  # reset if improve
            else:
                trigger_times += 1
                if trigger_times >= PATIENCE:
                    #print("Early stopping: loss not improved in {} epoches.".format(PATIENCE))
                    break
        
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
                 evaluator = LinkPredictionEvaluator(model, val_kg)
                 evaluator.evaluate(b_size=B_SIZE, verbose=False)
                 f_rank_true_heads = (evaluator.filt_rank_true_heads.float()**(-1)).mean()
                 f_rank_true_tails = (evaluator.filt_rank_true_tails.float()**(-1)).mean()
                 mr = (f_rank_true_heads + f_rank_true_tails).item() / 2
        #print(f" {mr:.2f} ")
        if mr > best_mrr:
            best_mrr = mr
            #print(f"Best MeanRank found {mr:.2f}")
            best_params = {
                "EMB_DIM": EMB_DIM,
                "LR": LR,
                "ETA": ETA,
                "N_EPOCHS": N_EPOCHS,
                "REG_TYPE": REG_TYPE,
                "P": P,
                "REG_CONST": REG_CONST,
            }
            best_model_state_dict = model.state_dict()

            
    return  best_params, best_mrr, best_model_state_dict



    
    
def grid_search_and_save_params(train_kg,val_kg):


    
    best_params, best_mr, best_model_state_dict = grid_search(PARAMS_GRID,BATCH_SIZE,train_kg,val_kg)
    
    #printing and saving important info
    print("=" * 40)
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("-" * 40)
    print(f"Best Mean Rank (MR): {round(best_mr,2)}")

    output_file = os.path.join(LOAD_DIR_MODEL, DATA_FILENAME[:-5]+"_best_params.json")

    Dataset_utils.check_for_folder(LOAD_DIR_MODEL) 

    save_to_json(output_file,best_params)
    print(f"Best params saved in : {output_file}")
    
    #uncomment to save the weights of the best model found
    #model_path = os.path.join(LOAD_DIR_MODEL, BEST_MODEL_FILENAME)
    #torch.save(best_model_state_dict, model_path)
    #print(f"Best models saved in: {model_path}")
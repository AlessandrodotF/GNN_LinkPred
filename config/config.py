SEED=0


DATASET_NAME = "PROV_network_test" #name of the folder containing data .json file
WORKING_DATASET_FILE= f"{DATASET_NAME}_clean.json" #!!!!! #USE {...}_clean.json if it has been preprocessed OTHERWISE {...}_no_clean.json

BATCH_SIZE_PARALLEL = 100 # parallel execution for the triplets creation 
TAXONOMY_OF_INTEREST = ["#Person"]#keep the list format

###################################################################################################
################################   DO NOT CHANGE   ################################################
DIR = f"dataset/{DATASET_NAME}/"
LOAD_DIR_MODEL = f"Saved_Models/ComplEx/{DATASET_NAME}/"
DATA_FILENAME=f"{DATASET_NAME}_FINAL_valid_triplets.json"
UNSEEN_TRIPLES = f"UNSEEN_{DATA_FILENAME}"
SAVE_TRIPLETS_SCORES = "statements_and_metrics_"+UNSEEN_TRIPLES[:-5]+".csv"
BEST_MODEL_FILENAME = "ComplEx_Best_GridSearch_"+DATA_FILENAME[:-5]+".pth"#save GS weights
MODEL_FILENAME = "ComplEx_main_model_"+DATA_FILENAME[:-5]+".pth"#save weights after training phase
###################################################################################################


#dataset parameters
ADD_RECIPROCAL_RELS = True
TEST_SIZE = 0.1# % of split respect to the full length of triplets file 
VAL_SIZE  = 0.1#  (they take in account the add_reciproc param) 

#select best model - GridSearch
PARAMS_GRID = {
     "EMB_DIM": [100],
     "LR": [1e-3],
     "ETA": [30],
     "N_EPOCHS": [25],
     "REG_TYPE": ["LP"],
     "P": [3],
     "REG_CONST": [1e-4,]}

#specific parameters for the training phase, 
#the others are loaded from GS results
PATIENCE=10 # (early stopping)
N_EPOCHS = 30
BATCH_SIZE = 500 #for train,val,test,inference and GS phase



#discovery.py NOT EXPERIMENTED at the moment
TOP_N=1
MAX_CANDIDATES=20000
STRATEGY='entity_frequency'
TARGET_REL='bought_from' 
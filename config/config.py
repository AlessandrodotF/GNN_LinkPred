
import yaml

class Config:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Attributi principali
        self.SEED = cfg.get("SEED", 0)
        self.DATASET_NAME = cfg["DATASET_NAME"]
        self.WORKING_DATASET_FILE = cfg["WORKING_DATASET_FILE"]

        self.BATCH_SIZE_PARALLEL = cfg["BATCH_SIZE_PARALLEL"]
        self.TAXONOMY_OF_INTEREST = cfg["TAXONOMY_OF_INTEREST"]

        # Percorsi derivati
        self.DIR = f"dataset/{self.DATASET_NAME}/"
        self.DATA_FILENAME = f"{self.DATASET_NAME}_FINAL_valid_triplets.json"
        self.UNSEEN_TRIPLES = f"UNSEEN_{self.DATA_FILENAME}"
        self.SAVE_TRIPLETS_SCORES = "statements_and_metrics_" + self.UNSEEN_TRIPLES[:-5] + ".csv"


        self.LOAD_DIR_MODEL = f"saved_models/ComplEx/{self.DATASET_NAME}/"
        self.BEST_MODEL_FOUND = "ComplEx_Best_GridSearch_" + self.DATA_FILENAME[:-5] + ".pth"
        self.MODEL_FILENAME = "ComplEx_main_model_" + self.DATA_FILENAME[:-5] + ".pth"

        # Altri parametri
        self.ADD_RECIPROCAL_RELS = cfg["ADD_RECIPROCAL_RELS"]
        self.TEST_SIZE = cfg["TEST_SIZE"]
        self.VAL_SIZE = cfg["VAL_SIZE"]

        self.PARAMS_GRID = cfg["PARAMS_GRID"]
        self.PATIENCE = cfg["PATIENCE"]
        self.N_EPOCHS = cfg["N_EPOCHS"]
        self.BATCH_SIZE = cfg["BATCH_SIZE"]

        self.TOP_N = cfg["TOP_N"]
        self.MAX_CANDIDATES = cfg["MAX_CANDIDATES"]
        self.strSTRATEGYategy = cfg["STRATEGY"]
        self.TARGET_REL = cfg["TARGET_REL"]

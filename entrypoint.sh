#!bin/bash

set -e #stop in case there are errors

#check if .yaml file exist (note spaces)
if [ "$#" -ne 2 ]; then
    echo "Input args should be only 2 $0 config_file.yaml raw_dataset.json"
    exit 1
else
    echo "Enough argument"
    CONFIG_PATH="$1"
    JSON_FILE="$2"
fi

#check if .yaml file exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: CONFIG_PATH not found: $CONFIG_PATH"
    exit 1
fi
#check if .json file exist
#-f check if it is a normal file not a directory for example
if [ ! -f "$JSON_FILE" ]; then 
    echo "Error: JSON file not found: $JSON_FILE"
    exit 1
fi

# Estrai il nome del dataset dal file YAML
DATASET_NAME=$(grep -E '^DATASET_NAME:' "$CONFIG_PATH" | awk '{gsub(/"/, "", $2); print $2}')
WORKING_DATASET_FILE=$(grep -E '^WORKING_DATASET_FILE:' "$CONFIG_PATH" | awk '{gsub(/"/, "", $2); print $2}')




# Check if  DATASET_NAME is found:
#-z check if a variable ie empty 
if [ -z "$DATASET_NAME" ]; then
    echo "Errore: Campo DATASET_NAME non trovato nel file YAML."
    exit 1
fi

# create folder
DEST_DIR="dataset/${DATASET_NAME}"


mkdir -p "$DEST_DIR"

# Copia il file JSON nella cartella con nome coerente
cp "$JSON_FILE" "${DEST_DIR}/$WORKING_DATASET_FILE"

#start the main.py using the given yaml file
python main.py --config "$CONFIG_PATH"
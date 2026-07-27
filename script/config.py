import os 
BASE_DIR =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
DATASET_DIR =os .path .join (BASE_DIR ,"dataset")
FEATURES_DIR =os .path .join (BASE_DIR ,"features")
HDF5_DATASET_PATH =os .path .join (BASE_DIR ,"dataset.h5")
MODELS_DIR =os .path .join (BASE_DIR ,"models")
INDEX_DIR =os .path .join (BASE_DIR ,"index")
DATASET_CSV_PATH =os .path .join (BASE_DIR ,"dataset.csv")
EXHAUSTED_CSV_PATH =os .path .join (BASE_DIR ,"exhausted_songs.csv")
MELODY_INDEX_PATH =os .path .join (BASE_DIR ,"melody_notes.h5")
for dir_path in [FEATURES_DIR ,MODELS_DIR ,INDEX_DIR ]:
    os .makedirs (dir_path ,exist_ok =True )
SAMPLE_RATE =22050 
SEGMENT_DURATION =15.0 
SEGMENT_HOP =5.0 
MIN_SEGMENT_DURATION =10.0 
BINS_PER_OCTAVE =24 
N_OCTAVES =7 
N_CQT_BINS =BINS_PER_OCTAVE *N_OCTAVES 
HOP_LENGTH =512 
MIN_NOTE ="C1"
FRAMES_PER_SEGMENT =int ((SEGMENT_DURATION *SAMPLE_RATE )/HOP_LENGTH )
EMBEDDING_DIM =256 
CNN_CHANNELS =[64 ,128 ,256 ]
LSTM_HIDDEN =128 
LSTM_LAYERS =2 
DROPOUT =0.2 
BOTTLENECK_DIM =192 
BATCH_SIZE =8 
ACCUMULATION_STEPS =16                                                                  
LEARNING_RATE =5e-5 
WEIGHT_DECAY =1e-4 
EPOCHS =50 
TEMPERATURE =0.07 
PATIENCE =10 
MIN_DELTA =1e-3 
SIMILARITY_THRESHOLD =0.7
TOP_K_RESULTS =100 
MIN_CONSECUTIVE_MATCHES =2 
import torch 
DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
import multiprocessing as _mp 
if _mp .current_process ().name =='MainProcess':
    print (f"[Config] Device: {DEVICE}")
    print (f"[Config] Segment: {SEGMENT_DURATION}s with {SEGMENT_HOP}s hop")
    print (f"[Config] Embedding dim: {EMBEDDING_DIM}")
    print (f"[Config] Model: Demucs-CQT (drum-removed; bass+other+vocals, {N_CQT_BINS} bins)")
    print (f"[Config] CNN: {CNN_CHANNELS}, LSTM: {LSTM_HIDDEN}x{LSTM_LAYERS}, Bottleneck: {BOTTLENECK_DIM}")
    print (f"[Config] NT-Xent Temperature: {TEMPERATURE}")

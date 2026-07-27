import os 
import sys 
import re 
import argparse 
import textwrap 
from concurrent .futures import ThreadPoolExecutor 
from typing import Dict ,List ,Optional ,Tuple 
import numpy as np 
from tqdm import tqdm 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import torch 
import torch .nn .functional as F 
import librosa 
try :
    from audio_utils import (
    normalize_query_audio ,cleanup_normalized_audio ,
    extract_melody_segment ,download_and_extract_proof_melody ,
    cleanup_proof_session_cache ,redetect_notes_from_audio ,
    prefetch_proof_songs 
    )
except ImportError :
    from script .audio_utils import (
    normalize_query_audio ,cleanup_normalized_audio ,
    extract_melody_segment ,download_and_extract_proof_melody ,
    cleanup_proof_session_cache ,redetect_notes_from_audio ,
    prefetch_proof_songs 
    )
try :
    from midi_utils import notes_to_wav ,is_synthesis_available ,generate_comparison_plot ,is_plot_available 
except ImportError :
    from script .midi_utils import notes_to_wav ,is_synthesis_available ,generate_comparison_plot ,is_plot_available 
try :
    from config import (
    DEVICE ,SAMPLE_RATE ,SEGMENT_DURATION ,SEGMENT_HOP ,
    HOP_LENGTH ,MODELS_DIR ,INDEX_DIR ,BINS_PER_OCTAVE ,MIN_NOTE ,
    HDF5_DATASET_PATH ,MELODY_INDEX_PATH ,
    )
    from model import MelodySimilarityModel 
    from extract_features import load_audio ,get_features_for_audio 
    from build_index import load_model ,load_index 
except ImportError :
    from script .config import (
    DEVICE ,SAMPLE_RATE ,SEGMENT_DURATION ,SEGMENT_HOP ,
    HOP_LENGTH ,MODELS_DIR ,INDEX_DIR ,BINS_PER_OCTAVE ,MIN_NOTE ,
    HDF5_DATASET_PATH ,MELODY_INDEX_PATH ,
    )
    from script .model import MelodySimilarityModel 
    from script .extract_features import load_audio ,get_features_for_audio 
    from script .build_index import load_model ,load_index 
try :
    import h5py 
    H5PY_AVAILABLE =True 
except ImportError :
    H5PY_AVAILABLE =False 

try :
    from demucs_utils import remove_drums as _remove_drums_fn 
    _DEMUCS_AVAILABLE =True 
except ImportError :
    try :
        from script .demucs_utils import remove_drums as _remove_drums_fn 
        _DEMUCS_AVAILABLE =True 
    except ImportError :
        _remove_drums_fn =None 
        _DEMUCS_AVAILABLE =False 
_NOTE_NAMES =["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_SOLFEGE =["Do","Do#","Re","Re#","Mi","Fa","Fa#","Sol","Sol#","La","La#","Si"]
def midi_to_note (midi :int )->str :
    midi =int (midi )
    if midi <0 or midi >127 :
        return "?"
    return _NOTE_NAMES [midi %12 ]+str (midi //12 -1 )
def midi_to_solfege (midi :int )->str :
    midi =int (midi )
    if midi <0 or midi >127 :
        return "?"
    return _SOLFEGE [midi %12 ]
def notes_to_str (notes :np .ndarray ,max_n :int =16 )->str :
    if len (notes )==0 :
        return "(tidak ada nada)"
    names =[midi_to_note (int (n ))for n in notes [:max_n ]]
    suffix =" ..."if len (notes )>max_n else ""
    return "  ".join (names )+suffix 
def notes_to_solfege_str (notes :np .ndarray ,max_n :int =16 )->str :
    if len (notes )==0 :
        return "(tidak ada nada)"
    sol =[midi_to_solfege (int (n ))for n in notes [:max_n ]]
    suffix =" ..."if len (notes )>max_n else ""
    return "  ".join (sol )+suffix 
def format_time(s: float) -> str:
    total_seconds = int(round(s))
    m, sec = divmod(total_seconds, 60)
    return f"{m}:{sec:02d}"
def format_range (start :float ,end :float )->str :
    return f"{format_time(start)}–{format_time(end)}"
_NOTE_SEMI ={
"C":0 ,"C#":1 ,"Db":1 ,"D":2 ,"D#":3 ,"Eb":3 ,
"E":4 ,"F":5 ,"F#":6 ,"Gb":6 ,"G":7 ,"G#":8 ,
"Ab":8 ,"A":9 ,"A#":10 ,"Bb":10 ,"B":11 ,
}
def key_to_shift (key_str :str )->int :
    if not key_str or key_str in ("Unknown",""):
        return 0 
    parts =key_str .split ()
    root =parts [0 ]
    semi =_NOTE_SEMI .get (root ,0 )
    return -semi 
def shift_notes_to_c (notes :np .ndarray ,key_str :str )->np .ndarray :
    shift =key_to_shift (key_str )
    if shift ==0 :
        return notes 
    return np .clip (notes .astype (np .int16 )+shift ,21 ,108 ).astype (np .int16 )
def lcs_length (a :np .ndarray ,b :np .ndarray ,tolerance :int =1 )->int :
    if len (a )==0 or len (b )==0 :
        return 0 
    m ,n =len (a ),len (b )
    prev =np .zeros (n +1 ,dtype =np .int32 )
    for i in range (m ):
        curr =np .zeros (n +1 ,dtype =np .int32 )
        for j in range (n ):
            if abs (int (a [i ])-int (b [j ]))<=tolerance :
                curr [j +1 ]=prev [j ]+1 
            else :
                curr [j +1 ]=max (prev [j +1 ],curr [j ])
        prev =curr 
    return int (prev [n ])
def melody_score (q_notes :np .ndarray ,m_notes :np .ndarray ,
q_key :str ="C Major",m_key :str ="C Major",
tolerance :int =1 ,
octave_invariant :bool =False )->float :
    q_shifted =shift_notes_to_c (q_notes ,q_key )
    m_shifted =shift_notes_to_c (m_notes ,m_key )
    if octave_invariant :
        q_shifted =q_shifted .astype (np .int16 )%12 
        m_shifted =m_shifted .astype (np .int16 )%12 
    if len (q_shifted )<3 or len (m_shifted )<3 :
        return 0.0 
    lcs =lcs_length (q_shifted ,m_shifted ,tolerance =tolerance )
    denom =len (q_shifted )
    raw =lcs /denom 
    if raw <0.40 and tolerance <2 :
        lcs2 =lcs_length (q_shifted ,m_shifted ,tolerance =2 )
        raw2 =lcs2 /denom 
        raw =max (raw ,raw2 *0.85 )
    CHANCE =0.15 
    if raw <=CHANCE :
        return 0.0 
    return (raw -CHANCE )/(1.0 -CHANCE )
def matched_note_count (q_notes :np .ndarray ,m_notes :np .ndarray ,
q_key :str ="C Major",m_key :str ="C Major",
tolerance :int =1 )->Tuple [int ,int ]:
    q_shifted =shift_notes_to_c (q_notes ,q_key )
    m_shifted =shift_notes_to_c (m_notes ,m_key )
    if len (q_shifted )==0 or len (m_shifted )==0 :
        return 0 ,0 
    return lcs_length (q_shifted ,m_shifted ,tolerance =tolerance ),max (len (q_shifted ),len (m_shifted ))
def all_window_matches (q_notes :np .ndarray ,m_notes :np .ndarray ,
q_key :str ="C Major",m_key :str ="C Major",
window :int =6 ,tolerance :int =1 ,
top_n :int =5 )->List [Dict ]:
    q_shifted =shift_notes_to_c (q_notes ,q_key )
    m_shifted =shift_notes_to_c (m_notes ,m_key )
    win =min (window ,len (q_shifted ),len (m_shifted ))
    if win <3 :
        return []
    all_wins :List [Dict ]=[]
    for q_start in range (len (q_shifted )-win +1 ):
        q_win =q_shifted [q_start :q_start +win ]
        q_orig =q_notes [q_start :q_start +win ]
        for m_start in range (len (m_shifted )-win +1 ):
            m_win =m_shifted [m_start :m_start +win ]
            m_orig =m_notes [m_start :m_start +win ]
            matches =sum (
            1 for a ,b in zip (q_win ,m_win )
            if abs (int (a )-int (b ))<=tolerance 
            )
            all_wins .append ({
            "q_start":q_start ,
            "q_end":q_start +win -1 ,
            "m_start":m_start ,
            "m_end":m_start +win -1 ,
            "matches":matches ,
            "window":win ,
            "ratio":matches /win ,
            "q_notes_orig":q_orig ,
            "m_notes_orig":m_orig ,
            "q_notes_solfege":q_win ,
            "m_notes_solfege":m_win ,
            })
    all_wins .sort (key =lambda x :(-x ["matches"],x ["q_start"],x ["m_start"]))
    selected :List [Dict ]=[]
    for w in all_wins :
        overlap =False 
        for s in selected :
            q_ov =not (w ["q_end"]<s ["q_start"]or w ["q_start"]>s ["q_end"])
            m_ov =not (w ["m_end"]<s ["m_start"]or w ["m_start"]>s ["m_end"])
            if q_ov and m_ov :
                overlap =True 
                break 
        if not overlap :
            selected .append (w )
        if len (selected )>=top_n :
            break 
    return selected 
def _simplify_note_seq (raw :List [int ],min_frames :int =3 )->np .ndarray :
    if not raw :
        return np .array ([],dtype =np .int8 )
    simp ,cur ,count =[],raw [0 ],1 
    for m in raw [1 :]:
        if m ==cur :
            count +=1 
        else :
            if count >=min_frames :
                simp .append (cur )
            cur ,count =m ,1 
    if count >=min_frames :
        simp .append (cur )
    return np .array (simp ,dtype =np .int8 )
def _monophonize (note_events :list )->list :
    if not note_events :
        return []
    events =sorted (note_events ,key =lambda e :(round (e [0 ],2 ),-e [3 ]))
    mono =[]
    current_end =-1.0 
    OVERLAP_TOLERANCE =0.05 
    for evt in events :
        start_t ,end_t ,pitch_midi ,amp =evt [0 ],evt [1 ],evt [2 ],evt [3 ]
        if start_t >=current_end -OVERLAP_TOLERANCE :
            midi =int (max (24 ,min (107 ,round (pitch_midi ))))
            mono .append (midi )
            current_end =end_t 
    return mono 
def extract_notes_basic_pitch (audio :np .ndarray )->np .ndarray :
    if len (audio )<int (SAMPLE_RATE *0.3 ):
        return np .array ([],dtype =np .int8 )
    try :
        import tempfile ,os as _os ,soundfile as _sf 
        import sys ,contextlib ,logging 
        logger =logging .getLogger ()
        old_level =logger .getEffectiveLevel ()
        logger .setLevel (logging .ERROR )
        try :
            with open (_os .devnull ,'w')as devnull :
                with contextlib .redirect_stdout (devnull ),contextlib .redirect_stderr (devnull ):
                    from basic_pitch .inference import predict as bp_predict 
                    from basic_pitch import ICASSP_2022_MODEL_PATH 
        finally :
            logger .setLevel (old_level )
        audio_f32 =np .clip (audio .astype (np .float32 ),-1.0 ,1.0 )
        fd ,tmp_wav =tempfile .mkstemp (suffix =".wav")
        _os .close (fd )
        try :
            _sf .write (tmp_wav ,audio_f32 ,SAMPLE_RATE )
            try :
                with open (_os .devnull ,'w')as devnull :
                    with contextlib .redirect_stdout (devnull ),contextlib .redirect_stderr (devnull ):
                        _ ,_ ,note_events =bp_predict (
                        tmp_wav ,
                        ICASSP_2022_MODEL_PATH ,
                        onset_threshold =0.5 ,
                        frame_threshold =0.3 ,
                        minimum_note_length =80.0 ,
                        minimum_frequency =float (librosa .note_to_hz ("C2")),
                        maximum_frequency =float (librosa .note_to_hz ("C6")),
                        melodia_trick =True ,
                        multiple_pitch_bends =False ,
                        )
            finally :
                logger .setLevel (old_level )
        finally :
            try :
                _os .remove (tmp_wav )
            except Exception :
                pass 
        if not note_events :
            return np .array ([],dtype =np .int8 )
        mono =_monophonize (note_events )
        if not mono :
            return np .array ([],dtype =np .int8 )
        deduped =[mono [0 ]]
        for n in mono [1 :]:
            if n !=deduped [-1 ]:
                deduped .append (n )
        return np .array (deduped ,dtype =np .int8 )
    except Exception as _bp_err :
        try :
            import torchcrepe 
            audio_t =torch .tensor (audio ,dtype =torch .float32 ).unsqueeze (0 ).to (DEVICE )
            f0 ,pd =torchcrepe .predict (
            audio_t ,SAMPLE_RATE ,HOP_LENGTH ,
            fmin =float (librosa .note_to_hz ("C2")),
            fmax =float (librosa .note_to_hz ("C6")),
            model ='tiny',return_periodicity =True ,
            batch_size =1024 ,device =DEVICE ,
            )
            f0 =f0 .squeeze (0 ).cpu ().numpy ()
            pd =pd .squeeze (0 ).cpu ().numpy ()
            raw =[]
            for pitch ,v in zip (f0 ,pd ):
                if v <0.3 or np .isnan (pitch )or pitch <=0 :
                    continue 
                raw .append (int (max (24 ,min (107 ,round (librosa .hz_to_midi (float (pitch )))))))
            return _simplify_note_seq (raw ,min_frames =3 )
        except Exception :
            return np .array ([],dtype =np .int8 )
extract_notes_crepe =extract_notes_basic_pitch 
def melody_h5_stats ()->Tuple [bool ,int ]:
    if not H5PY_AVAILABLE or not os .path .exists (MELODY_INDEX_PATH ):
        return False ,0 
    try :
        with h5py .File (MELODY_INDEX_PATH ,"r")as h5 :
            return True ,len (h5 .keys ())
    except Exception :
        return False ,0 
def embedding_prefilter (query_embs :np .ndarray ,index :Dict ,
n_candidates :int )->List [List [Dict ]]:
    meta =index ["metadata"]
    db_full =np .array (index ["embeddings"],dtype =np .float32 )
                                                                               
    song_embs_db =index .get ("song_embeddings")
    song_meta_db =index .get ("song_metadata")
    if (song_embs_db is not None and song_meta_db
            and len (song_embs_db )>0 ):
        q_np =query_embs .astype (np .float32 )
        _nq =np .linalg .norm (q_np ,axis =1 ,keepdims =True )
        _nq [_nq ==0 ]=1e-8
        q_np =q_np /_nq
                                                                             
        song_sims_mat =song_embs_db @q_np .T                          
        song_sims =(0.5 *song_sims_mat .max (axis =1 )
                   +0.5 *song_sims_mat .mean (axis =1 ))
        pre_k =min (max (n_candidates *5 ,300 ),len (song_meta_db ))
        top_song_idxs =np .argpartition (song_sims ,-pre_k )[-pre_k :]
        allowed_sv ={
        f"{song_meta_db [i ]['song_id']}|{song_meta_db [i ]['version']}"
        for i in top_song_idxs
        }
        keep =np .array ([
        f"{m ['song_id']}|{m ['version']}"in allowed_sv
        for m in meta
        ],dtype =bool )
        db_np =db_full [keep ]
        filtered_meta =[meta [i ]for i ,k in enumerate (keep )if k ]
    else :
                                                                            
        db_np =db_full
        filtered_meta =meta
                                                                              
    q_t =torch .tensor (query_embs ,dtype =torch .float32 ).to (DEVICE )
    q_t =F .normalize (q_t ,dim =-1 )
    norms =np .linalg .norm (db_np ,axis =1 ,keepdims =True )
    norms [norms ==0 ]=1e-8
    db_np =db_np /norms
    db_t =torch .from_numpy (db_np ).to (DEVICE )
    with torch .no_grad ():
        sims =(q_t @db_t .T ).cpu ().numpy ()
    Nq =sims .shape [0 ]
    search_k =min (max (n_candidates *250 ,20000 ),sims .shape [1 ])
    per_query :List [List [Dict ]]=[]
    for q_idx in range (Nq ):
        top_idxs =np .argpartition (sims [q_idx ],-search_k )[-search_k :]
        top_idxs =top_idxs [np .argsort (sims [q_idx ,top_idxs ])[::-1 ]]
        best_per_song :Dict [str ,Dict ]={}
        for db_idx in top_idxs :
            m =filtered_meta [db_idx ]
            song_key =f"{m ['song_id']}|{m ['version']}"
            sim_val =float (sims [q_idx ,db_idx ])
            if song_key not in best_per_song or sim_val >best_per_song [song_key ]["embed_sim"]:
                best_per_song [song_key ]={
                "song_id":m ["song_id"],
                "version":m ["version"],
                "seg_idx":m ["segment_idx"],
                "start_time":m ["start_time"],
                "end_time":m ["end_time"],
                "global_key":m .get ("global_key","Unknown"),
                "embed_sim":sim_val ,
                }
        cands =sorted (best_per_song .values (),key =lambda x :x ["embed_sim"],reverse =True )
        per_query .append (cands [:n_candidates ])
    return per_query 
def load_song_metadata ()->Dict :
    import csv 
    meta ={}
    csv_path =os .path .join (
    os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),
    "dataset.csv"
    )
    if not os .path .exists (csv_path ):
        return meta 
    with open (csv_path ,encoding ="utf-8")as f :
        for row in csv .DictReader (f ):
            sid =row .get ("song_id","").strip ()
            if not sid :
                continue 
            ver =row .get ("version","").strip ().lower ()
            artist =row .get ("artist","").strip ()
            title =row .get ("title","").strip ()
            keys ={sid }
            stripped =sid .lstrip ("0")or "0"
            keys .add (stripped )
            try :
                keys .add (str (int (sid )).zfill (3 ))
            except ValueError :
                pass 
            for k in keys :
                if k not in meta :
                    meta [k ]={
                    "song_id":sid ,
                    "title":title ,
                    "artist":artist ,
                    "artist_original":"Unknown",
                    "artist_cover":"Unknown",
                    }
                if title :
                    meta [k ]["title"]=title 
                if ver =="original":
                    meta [k ]["artist_original"]=artist 
                    meta [k ]["artist"]=artist 
                elif ver =="cover":
                    meta [k ]["artist_cover"]=artist 
    return meta 
def is_youtube_url (text :str )->bool :
    patterns =[
    r'(https?://)?(www\.)?youtube\.com/watch\?v=',
    r'(https?://)?(www\.)?youtu\.be/',
    r'(https?://)?(www\.)?youtube\.com/shorts/',
    ]
    return any (re .search (p ,text )for p in patterns )
def download_youtube (url :str ,output_dir :Optional [str ]=None )->Optional [str ]:
    try :
        import yt_dlp 
    except ImportError :
        print ("\u274c yt-dlp belum terinstall. Jalankan: pip install yt-dlp")
        return None 
    if output_dir is None :
        output_dir =os .path .join (
        os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),'downloads'
        )
    os .makedirs (output_dir ,exist_ok =True )
                                                                           
    import time as _t
    unique_name =f"query_{int(_t.time()*1000)}"
    base_path =os .path .join (output_dir ,unique_name )
    output_wav =f"{base_path}.wav"
    print (f"\n\U0001f4e5 Downloading from YouTube \u2192 {output_dir}...")
    ydl_opts ={
    'format':'bestaudio/best',
    'postprocessors':[{
    'key':'FFmpegExtractAudio',
    'preferredcodec':'wav',
    'preferredquality':'0',
    }],
    'quiet':False ,
    'no_warnings':False ,
    'retries':5 ,
    'socket_timeout':120 ,
    'nocheckcertificate':True ,
    'extractor_args':{'youtube':{'player_client':['android','ios','web']}},
    'outtmpl':f'{base_path}.%(ext)s',
    }
    try :
        with yt_dlp .YoutubeDL (ydl_opts )as ydl :
            ydl .download ([url ])
        if os .path .exists (output_wav ):
            print (f"\u2705 Downloaded: {output_wav}")
            return output_wav 
        for ext in ['wav','mp3','opus','m4a','webm']:
            alt =f"{base_path}.{ext}"
            if os .path .exists (alt ):
                print (f"\u2705 Downloaded: {alt}")
                return alt 
        print (f"\u274c File audio tidak ditemukan di {output_dir} setelah download")
        return None 
    except Exception as e :
        print (f"\u274c Download error: {e}")
        return None 
def melody_query_core (
input_path :str ,
model ,
index :Dict ,
meta_db :Dict ,
top_k :int =5 ,
n_candidates :int =80 ,
min_score :float =0.45 ,
progress_cb =None ,
)->Dict :
    import time as _time
    from query import extract_query_embeddings 
    def _step (msg :str ):
        if progress_cb :
            progress_cb (msg )
    warnings :List [str ]=[]
    _step ("Memuat audio...")
    raw_audio =load_audio (input_path )
    if raw_audio is None :
        raise RuntimeError ("Tidak bisa load file audio.")
    try :
        _tempo ,_ =librosa .beat .beat_track (y =raw_audio ,sr =SAMPLE_RATE )
        query_bpm =float (np .atleast_1d (_tempo )[0 ])
    except Exception :
        query_bpm =0.0 

    _step ("Ekstrak fitur CQT (drum-removed, via Demucs internal)...")
    _t_cqt =_time .perf_counter ()
    feat =get_features_for_audio (raw_audio )
    _cqt_sec =round (_time .perf_counter ()-_t_cqt ,2 )
    if feat is None :
        raise RuntimeError ("Gagal extract features audio.")
    cqt_feats =feat .get ("harmonic_cqt")
    if cqt_feats is None :
        cqt_feats =feat .get ("cqt_features")
    seg_meta =feat ["metadata"]
    query_key =feat .get ("global_key","Unknown")
    key_conf =feat .get ("key_confidence",0.0 )
    if key_conf <0.10 :
        warnings .append (
        "Kunci nada tidak terdeteksi dengan yakin — hasil mungkin kurang akurat."
        )
    for i ,seg in enumerate (seg_meta ):
        seg ["nn_features"]=cqt_feats [i ]
        seg .setdefault ("start_time",i *SEGMENT_HOP )
        seg .setdefault ("end_time",i *SEGMENT_HOP +SEGMENT_DURATION )
    _step ("Stage 1 — Embedding prefilter (multi-key)...")
    _t_infer =_time .perf_counter ()
    _bins_per_semitone =BINS_PER_OCTAVE //12 
    key_shifts_semi =[0 ,1 ,-1 ,2 ,-2 ,3 ,-3 ,4 ,-4 ,5 ,-5 ]
    per_query_merged :List [Dict [str ,Dict ]]=[{}for _ in seg_meta ]
    for k_shift in key_shifts_semi :
        if k_shift ==0 :
            shifted_feats =cqt_feats 
        else :
            shifted_feats =np .roll (cqt_feats ,k_shift *_bins_per_semitone ,axis =1 )
        seg_meta_shifted =[dict (s )for s in seg_meta ]
        for i ,seg in enumerate (seg_meta_shifted ):
            seg ["nn_features"]=shifted_feats [i ]
        q_embs_shifted =extract_query_embeddings (model ,seg_meta_shifted )
        cands_shifted =embedding_prefilter (q_embs_shifted ,index ,
        n_candidates =n_candidates )
        for i ,cands in enumerate (cands_shifted ):
            for cand in cands :
                song_key =f"{cand['song_id']}|{cand['version']}"
                existing =per_query_merged [i ].get (song_key )
                if existing is None or cand ["embed_sim"]>existing ["embed_sim"]:
                    per_query_merged [i ][song_key ]=cand 
    _top_n =max (n_candidates ,150 )
    per_query_cands =[
    sorted (d .values (),key =lambda x :x ["embed_sim"],reverse =True )[:_top_n ]
    for d in per_query_merged 
    ]
    for i ,seg in enumerate (seg_meta ):
        seg ["nn_features"]=cqt_feats [i ]
    _step ("Stage 1b — Re-ranking embed_sim (k_shift=0, cegah false-positive inflasi)...")
    q_embs_k0 =extract_query_embeddings (model ,seg_meta )
    cands_k0_per =embedding_prefilter (q_embs_k0 ,index ,n_candidates =_top_n )
    for q_i in range (len (per_query_cands )):
        k0_lookup ={
        f"{c['song_id']}|{c['version']}":c ["embed_sim"]
        for c in cands_k0_per [q_i ]
        }
        for cand in per_query_cands [q_i ]:
            sk =f"{cand['song_id']}|{cand['version']}"
            if sk in k0_lookup :
                cand ["embed_sim"]=max (cand ["embed_sim"],k0_lookup [sk ])
            else :
                cand ["embed_sim"]*=0.95 
        per_query_cands [q_i ].sort (key =lambda x :x ["embed_sim"],reverse =True )
                                                                                  
    _step ("Stage 2 — Menyiapkan field display (pure embedding mode)...")
    _empty =np .array ([],dtype =np .int8 )
    for _cands in per_query_cands :
        for _cand in _cands :
            _cand ["q_notes"]=_empty 
            _cand ["m_notes"]=_empty 
            _cand ["win_sim"]=0.0 
            _cand ["lcs_sim"]=0.0 
            _cand ["melody"]=_cand ["embed_sim"]
                                                                                 
    _step ("Finalisasi hasil...")
    results_per_seg :Dict [int ,List [Dict ]]={}
    for i ,cands in enumerate (per_query_cands ):
        scored =sorted (cands ,key =lambda x :x ["embed_sim"],reverse =True )
        results_per_seg [i ]=scored [:max (top_k ,20 )]
    _infer_sec =round (_time .perf_counter ()-_t_infer ,2 )
    return {
    "results_per_seg":results_per_seg ,
    "seg_meta":seg_meta ,
    "query_key":query_key ,
    "key_conf":key_conf ,
    "query_bpm":round (query_bpm ,1 ),
    "query_notes":{},
    "warnings":warnings ,
    "cqt_sec":_cqt_sec ,
    "infer_sec":_infer_sec ,
    }
def melody_first_query (input_path :str ,top_k :int =5 ,
n_candidates :int =80 ,
model_path =None ,
min_melody_score :float =0.45 ):
    W =70 
    _original_input =input_path 
    _normalized_tmp =None 
    if is_youtube_url (input_path ):
        downloaded =download_youtube (input_path )
        if downloaded is None :
            print ("Gagal download audio dari YouTube.")
            return 
        input_path =downloaded 
    else :
        _normalized_tmp =normalize_query_audio (input_path )
        if _normalized_tmp !=input_path :
            input_path =_normalized_tmp 
    print ("\n"+"="*W )
    print ("  MELODY-FIRST SIMILARITY SEARCH")
    print ("  Ranking utama: LCS note-interval comparison (bukan embedding)")
    print ("="*W )
    h5_ok ,h5_songs =melody_h5_stats ()
    if h5_ok :
        print (f"  melody_notes.h5 : OK  {h5_songs} lagu precomputed")
    else :
        print ("  melody_notes.h5 : Belum tersedia")
        print ("                    Jalankan build_melody_index.py untuk mempercepat.")
    print (f"\n  Kandidat embedding : {n_candidates} per segment")
    print (f"  Top-K hasil        : {top_k}")
    print (f"  Min melody score   : {min_melody_score*100:.0f}%\n")
    model =load_model (model_path )
    index =load_index ()
    print (f"  Index : {len(index['embeddings'])} segments dari database")
    meta_db =load_song_metadata ()
    def _cli_progress (msg :str ):
        print (f"  -> {msg}")
    core =melody_query_core (
    input_path ,model ,index ,meta_db ,
    top_k =top_k ,n_candidates =n_candidates ,min_score =min_melody_score ,
    progress_cb =_cli_progress ,
    )
    for w in core ["warnings"]:
        print (f"  WARN: {w}")
    _display (core ["results_per_seg"],core ["seg_meta"],core ["query_key"],
    _original_input ,meta_db ,top_k ,min_melody_score ,W )
    ask =input ("\nGenerate melody proof files? (y/n) [n]: ").strip ().lower ()
    if ask =="y":
        _generate_proof (core ["results_per_seg"],core ["seg_meta"],
        core ["query_key"],_original_input ,meta_db ,
        min_melody_score )
    if _normalized_tmp is not None :
        cleanup_normalized_audio (_normalized_tmp ,_original_input )
def _generate_proof (results_per_seg :Dict [int ,List [Dict ]],
seg_meta :List [Dict ],
query_key :str ,
input_path :str ,
meta_db :Dict ,
min_score :float ):
    import re 
    def _safe_name (path :str )->str :
        yt =re .search (r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})',path )
        if yt :
            return f"yt_{yt.group(1)}"
        name =os .path .splitext (os .path .basename (path ))[0 ]
        return re .sub (r'[<>:"/\\|?*\s]+','_',name ).strip ('_')or "query"
    query_name =_safe_name (input_path )
    root_dir =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
    proof_root =os .path .join (root_dir ,"proof",query_name )
    os .makedirs (proof_root ,exist_ok =True )
    print (f"\n📂 Generating proof in: {proof_root}")
    _wav_available =is_synthesis_available ()
    _plot_available =is_plot_available ()
    _proof_dl_cache :dict ={}
    _all_dl_tasks :list =[]
    for _seg_results in results_per_seg .values ():
        for _r in _seg_results :
            if _r .get ("melody",0 )<min_score :
                continue 
            _sid =str (_r .get ("song_id",""))
            _ver =str (_r .get ("version","")).lower ()
            if _sid and _ver :
                _all_dl_tasks .append ((_sid ,_ver ))
    if _all_dl_tasks :
        prefetch_proof_songs (_all_dl_tasks ,_proof_dl_cache ,max_workers =4 )
    def _wrap (tokens :list ,width :int =54 )->list :
        out ,cur =[],""
        for t in tokens :
            chunk =("  "+t )if cur else t 
            if cur and len (cur )+len (chunk )>width :
                out .append (cur )
                cur =t 
            else :
                cur =cur +chunk 
        if cur :
            out .append (cur )
        return out or [""]
    def _score_label (pct :float )->str :
        if pct >=80 :return "🔥 Sangat Mirip"
        if pct >=60 :return "✅ Mirip"
        return "⚠️  Kurang Mirip"
    total_segs =0 
    for i ,seg in enumerate (seg_meta ):
        results =results_per_seg .get (i ,[])
        good =[r for r in results if r ["melody"]>=min_score ]
        if not good :
            continue 
        q_start =seg .get ("start_time",i *SEGMENT_HOP )
        q_end =seg .get ("end_time",q_start +SEGMENT_DURATION )
        q_notes =results [0 ]["q_notes"]if results else np .array ([],dtype =np .int8 )
        q_time =f"{format_time(q_start)}-{format_time(q_end)}"
        ts =q_time .replace (':','').replace ('-','_')
        seg_dir =os .path .join (proof_root ,f"seg{i+1:02d}_{ts}")
        os .makedirs (seg_dir ,exist_ok =True )
        _q_mel_wav =os .path .join (seg_dir ,"query_melody.wav")
        _q_real_wav =os .path .join (seg_dir ,"query_real.wav")
        if _wav_available :
            notes_to_wav (q_notes ,_q_mel_wav )
        _q_real_ok =extract_melody_segment (input_path ,q_start ,q_end ,_q_real_wav )
        q_sol_tokens =notes_to_solfege_str (
        shift_notes_to_c (q_notes ,query_key ),max_n =9999 
        ).split ()
        fpath =os .path .join (seg_dir ,"proof.txt")
        with open (fpath ,"w",encoding ="utf-8")as f :
            f .write ("╔"+"═"*54 +"╗\n")
            f .write ("║   LAPORAN KEMIRIPAN MELODI — Melody-First System   ║\n")
            f .write ("╚"+"═"*54 +"╝\n\n")
            f .write (f"File audio    : {os.path.basename(input_path)}\n")
            f .write (f"Segmen waktu  : {q_time.replace('-', ' – ')}"
            f" (durasi ±15 detik)\n")
            f .write (f"Tangga nada   : {query_key}\n")
            f .write (f"Nada di segmen: {len(q_notes)} nada terdeteksi\n\n")
            f .write ("File yang bisa kamu buka:\n")
            if _wav_available :
                f .write ("  🎹 query_melody.wav — melodi kamu, dimainkan sebagai piano\n")
            if _q_real_ok :
                f .write ("  🎵 query_real.wav   — melodi asli langsung dari audio kamu\n")
            f .write ("\n")
            f .write ("Urutan solfege yang terdeteksi dari audio kamu:\n")
            for _sl in _wrap (q_sol_tokens ):
                f .write (f"  {_sl}\n")
            f .write ("\n")
            f .write ("─"*56 +"\n")
            f .write (f"DITEMUKAN {len(good)} LAGU YANG COCOK (skor ≥ {min_score*100:.0f}%)\n")
            f .write ("─"*56 +"\n\n")
            for rank ,res in enumerate (good ,start =1 ):
                s_id =res ["song_id"]
                s_meta =meta_db .get (s_id ,{})
                title =s_meta .get ("title",f"Song {s_id}")
                ver =res ["version"]
                _ao =s_meta .get ("artist_original","Unknown")
                _ac =s_meta .get ("artist_cover","Unknown")
                artist =(_ac if ver =="cover"else _ao )
                if not artist or artist =="Unknown":
                    artist =s_meta .get ("artist","Unknown")
                ver_label ="versi cover"if ver =="cover"else "versi asli"
                m_notes =res ["m_notes"]
                m_start =res ["start_time"]
                m_end =res ["end_time"]
                m_time =f"{format_time(m_start)}-{format_time(m_end)}"
                mel_pct =res ["melody"]*100 
                emb_pct =res ["embed_sim"]*100 
                m_key =res .get ("global_key","C Major")
                lcs_n ,lcs_d =matched_note_count (q_notes ,m_notes ,
                q_key =query_key ,m_key =m_key )
                _m_mel_wav =os .path .join (seg_dir ,f"rank{rank}_melody.wav")
                _m_real_wav =os .path .join (seg_dir ,f"rank{rank}_real.wav")
                if _wav_available :
                    notes_to_wav (m_notes ,_m_mel_wav )
                _m_real_ok =download_and_extract_proof_melody (
                s_id ,ver ,m_start ,m_end ,_m_real_wav ,
                session_cache =_proof_dl_cache 
                )
                _png_path =os .path .join (seg_dir ,f"rank{rank}_pitch.png")
                _png_ok =False 
                if _plot_available and len (q_notes )>0 and len (m_notes )>0 :
                    _t =f"Perbandingan Melodi — Query vs #{rank}: {title}"
                    _png_ok =generate_comparison_plot (
                    q_notes ,m_notes ,query_key ,m_key ,_png_path ,title =_t 
                    )
                medal =["🥇","🥈","🥉"," 4."," 5."," 6."," 7."," 8."," 9.","10."][
                min (rank -1 ,9 )]
                f .write ("═"*56 +"\n")
                f .write (f" {medal} URUTAN #{rank}: {title.upper()}\n")
                f .write (f"     Artis  : {artist} ({ver_label})\n")
                f .write ("═"*56 +"\n\n")
                f .write (f"  Waktu di lagu    : {m_time.replace('-', ' – ')}\n")
                f .write (f"  Tangga nada lagu : {m_key}\n\n")
                f .write ("  SKOR KEMIRIPAN:\n")
                f .write (f"    Melodi   : {mel_pct:.0f}%  {_score_label(mel_pct)}\n")
                f .write (f"    Embedding: {emb_pct:.0f}%  {_score_label(emb_pct)}\n")
                if lcs_d >0 :
                    f .write (f"    Detail   : {lcs_n} dari {lcs_d} nada cocok"
                    f" ({lcs_n/lcs_d*100:.0f}%)\n")
                f .write ("\n")
                f .write ("  FILE UNTUK DIBANDINGKAN:\n")
                if _wav_available :
                    f .write (f"    🎹 rank{rank}_melody.wav  — melodi lagu ini (piano)\n")
                if _m_real_ok :
                    f .write (f"    🎵 rank{rank}_real.wav    — melodi asli dari dataset\n")
                if _q_real_ok and _m_real_ok :
                    f .write (f"\n    ► Dengarkan query_real.wav dan rank{rank}_real.wav\n")
                    f .write ("      Kalau keduanya terdengar mirip → sistem sudah benar!\n")
                if _png_ok :
                    f .write (f"\n    🖼  rank{rank}_pitch.png  — grafik pola naik-turun\n")
                    f .write ("       melodi kamu vs melodi lagu (buka gambar ini)\n")
                f .write ("\n")
                if _m_real_ok :
                    re_pct ,re_n ,re_tot ,re_notes =redetect_notes_from_audio (
                    _m_real_wav ,m_notes ,tolerance =1 
                    )
                    if re_pct is not None :
                        _rf ="✓ Terbukti akurat"if re_pct >=60 else "△ Sebagian cocok (noise HPSS)"
                        f .write (f"  VERIFIKASI NADA ({_rf}):\n")
                        f .write (f"    {re_n}/{re_tot} nada tersimpan terdeteksi ulang"
                        f" dari audio asli = {re_pct:.0f}%\n")
                        f .write ("    → Ini membuktikan nada yang ditampilkan\n")
                        f .write ("      memang sesuai dengan audio aslinya\n\n")
                        _st =notes_to_solfege_str (
                        shift_notes_to_c (m_notes ,m_key ),max_n =9999 
                        ).split ()
                        f .write (f"    Tersimpan  ({len(_st):>2} nada):\n")
                        for _sl in _wrap (_st ):
                            f .write (f"      {_sl}\n")
                        if re_notes :
                            _rd =notes_to_solfege_str (
                            shift_notes_to_c (
                            np .array (re_notes ,dtype =np .int8 ),m_key 
                            ),max_n =9999 
                            ).split ()
                            f .write (f"    Re-detected ({len(_rd):>2} nada):\n")
                            for _sl in _wrap (_rd ):
                                f .write (f"      {_sl}\n")
                        f .write ("\n")
                win_details =all_window_matches (
                q_notes ,m_notes ,q_key =query_key ,m_key =m_key ,
                window =12 ,tolerance =1 ,top_n =5 
                )
                if win_details :
                    w0 =win_details [0 ]
                    best_pct =w0 ["ratio"]*100 
                    f .write ("  KECOCOKAN POLA MELODI (per 12 nada):\n")
                    f .write (f"    Terbaik: {w0['matches']}/12 nada = {best_pct:.0f}%"
                    f"  {'★ Sangat baik' if best_pct >= 80 else '✓ Baik'}\n")
                    f .write ("    Kamu  : "
                    +"  ".join (midi_to_solfege (int (n ))
                    for n in w0 ["q_notes_solfege"])+"\n")
                    f .write ("    Lagu  : "
                    +"  ".join (midi_to_solfege (int (n ))
                    for n in w0 ["m_notes_solfege"])+"\n")
                    f .write ("    (Do = nada dasar lagu; ±1 semitone dianggap cocok)\n")
                    if len (win_details )>1 :
                        f .write ("\n    Semua window:\n")
                        for wi ,wd in enumerate (win_details ):
                            pct =wd ["ratio"]*100 
                            star =" ★"if wi ==0 else "  "
                            f .write (f"    {star} #{wi+1}: "
                            f"q[{wd['q_start']:>2}..{wd['q_end']:>2}] ↔ "
                            f"m[{wd['m_start']:>2}..{wd['m_end']:>2}] "
                            f"{wd['matches']}/12 ({pct:.0f}%)\n")
                f .write ("\n")
        total_segs +=1 
    print (f"   ✅ {total_segs} segmen proof tersimpan → {os.path.abspath(proof_root)}")
    cleanup_proof_session_cache (_proof_dl_cache )
    if is_synthesis_available ():
        print ("   🎵 File WAV tersimpan — klik dua kali untuk diputar")
    else :
        print ("   ⚠️  File audio dilewati (pip install scipy untuk mengaktifkan)")
_MEDALS =["🥇","🥈","🥉"," 4."," 5."," 6."," 7."," 8."," 9.","10."]
def _sim_tag (score :float )->str :
    if score >=0.70 :
        return "🔥 SANGAT TINGGI"
    elif score >=0.50 :
        return "✅ TINGGI      "
    elif score >=0.30 :
        return "🔶 SEDANG      "
    else :
        return "🔹 RENDAH      "
def _display (results_per_seg :Dict [int ,List [Dict ]],
seg_meta :List [Dict ],
query_key :str ,
input_path :str ,
meta_db :Dict ,
top_k :int ,
min_score :float ,
W :int ):
    print ("\n"+"═"*W )
    print (f"  HASIL  --  {os.path.basename(input_path)}")
    print (f"  Kunci query : {query_key}")
    print (f"  Ranking     : Embedding similarity (primary) | Melody score (bukti)")
    print (f"  Format nada : C4 = C tengah (do bawah), A4 = 440 Hz")
    print ("═"*W )
    total_shown =0 
    for i ,seg in enumerate (seg_meta ):
        results =results_per_seg .get (i ,[])
        good =[r for r in results if r ["embed_sim"]>=min_score ]
        if not good :
            continue 
        q_start =seg .get ("start_time",i *SEGMENT_HOP )
        q_end =seg .get ("end_time",q_start +SEGMENT_DURATION )
        q_notes =results [0 ]["q_notes"]if results else np .array ([],dtype =np .int8 )
        print (f"\n{'─' * W}")
        print (f"  🎵  Query [{format_range(q_start, q_end)}]")
        if len (q_notes )>0 :
            print (f"      Notes   : {notes_to_str(q_notes)}")
            print (f"      Solfege : {notes_to_solfege_str(shift_notes_to_c(q_notes, query_key))}")
            print (f"      ({len(q_notes)} nada terdeteksi)")
        print ()
        rank =0 
        for res in good [:top_k ]:
            if res ["embed_sim"]<min_score :
                continue 
            s_id =res ["song_id"]
            s_meta =meta_db .get (s_id ,{})
            title =s_meta .get ("title",f"Song {s_id}")
            ver =res ["version"]
            _ao =s_meta .get ("artist_original","Unknown")
            _ac =s_meta .get ("artist_cover","Unknown")
            artist =(_ac if ver =="cover"else _ao )
            if not artist or artist =="Unknown":
                artist =s_meta .get ("artist","Unknown")
            ver_ico ="cover"if ver =="cover"else "original"
            m_rng =format_range (res ["start_time"],res ["end_time"])
            m_notes =res ["m_notes"]
            m_key =res .get ("global_key","C Major")
            medal =_MEDALS [rank ]if rank <len (_MEDALS )else f"{rank+1:2d}."
            win_pct =res ["win_sim"]*100 
            lcs_pct =res ["lcs_sim"]*100 
            emb_pct =res ["embed_sim"]*100 
            print (f"  {medal}  {title}")
            print (f"        {artist}  |  {ver_ico}  |  [{m_rng}]  |  key: {m_key}")
            print (f"        Embed (CNN+BiLSTM) : {emb_pct:5.1f}%  << SKOR UTAMA")
            print ()
            if len (q_notes )>=3 and len (m_notes )>=3 :
                win_tag =_sim_tag (res ["win_sim"])
                print (f"        > WINDOW (8-nada berurutan): {win_pct:.1f}%  {win_tag}")
                win_details =all_window_matches (
                q_notes ,m_notes ,q_key =query_key ,m_key =m_key ,
                window =8 ,tolerance =1 ,top_n =3 
                )
                if win_details :
                    w0 =win_details [0 ]
                    print (f"          Window terbaik: "
                    f"q[{w0['q_start']:>2}..{w0['q_end']:>2}] "
                    f"m[{w0['m_start']:>2}..{w0['m_end']:>2}]  "
                    f"{w0['matches']}/8 nada cocok ({w0['ratio']*100:.0f}%)")
                    print (f"          Query  : "
                    +"  ".join (midi_to_solfege (int (n ))for n in w0 ["q_notes_solfege"]))
                    print (f"          Match  : "
                    +"  ".join (midi_to_solfege (int (n ))for n in w0 ["m_notes_solfege"]))
                    if len (win_details )>1 :
                        for wi ,wd in enumerate (win_details [1 :],start =2 ):
                            pct =wd ["ratio"]*100 
                            print (f"          Window #{wi}: "
                            f"q[{wd['q_start']:>2}..{wd['q_end']:>2}] "
                            f"m[{wd['m_start']:>2}..{wd['m_end']:>2}]  "
                            f"{wd['matches']}/8 = {pct:.0f}%")
                else :
                    print (f"          (tidak ada window 8-nada yang cocok)")
                print ()
            lcs_n ,lcs_d =matched_note_count (
            q_notes ,m_notes ,q_key =query_key ,m_key =m_key 
            )
            if lcs_d >0 :
                lcs_tag =_sim_tag (res ["lcs_sim"])
                print (f"        > LCS (nada cocok tersebar): {lcs_pct:.1f}%  {lcs_tag}")
                print (f"          {lcs_n} dari {lcs_d} nada cocok = {lcs_n/lcs_d*100:.0f}%")
                print (f"          Nada match : {notes_to_str(m_notes)}")
                print (f"          Solfege    : "
                f"{notes_to_solfege_str(shift_notes_to_c(m_notes, m_key))}")
                print ()
            rank +=1 
            total_shown +=1 
    if total_shown ==0 :
        print (f"\n  Tidak ada hasil dengan embed_sim >= "
        f"{min_score*100:.0f}% ditemukan.")
        print ("       Coba turunkan --min-melody atau tambah --candidates.")
    else :
        print ("-"*W )
        print ("\n  CARA BACA HASIL:")
        print ("      Embed  : skor CNN+BiLSTM embedding -- RANKING UTAMA")
        print ("      WINDOW : 8 nada berurutan yang sama -- bukti terkuat")
        print ("      LCS    : nada sama tersebar -- relevan untuk cover song")
        print ()
def _interactive_mode ():
    W =70 
    print ("\n"+"═"*W )
    print ("  🎼  MELODY-FIRST — MODE INTERAKTIF")
    print ("  Ketik path file audio, YouTube URL, atau 'q' untuk keluar.")
    print ("  Tekan Enter untuk menggunakan nilai default [dalam kurung].")
    print ("═"*W )
    top_k =5 
    n_candidates =80 
    min_melody =0.60 
    model_path =None 
    while True :
        print ()
        raw =input ("🎵 Audio / URL  (q=keluar, s=settings): ").strip ()
        if raw .lower ()in ("q","quit","exit",""):
            if raw =="":
                continue 
            print ("\n👋 Sampai jumpa!")
            break 
        if raw .lower ()in ("s","settings","setting"):
            print (f"\n┌─ Pengaturan Saat Ini {'─'*46}┐")
            print (f"│  1. Top-K hasil per segment : {top_k:<5}                       │")
            print (f"│  2. Kandidat embedding      : {n_candidates:<5}                       │")
            print (f"│  3. Min melody score        : {min_melody*100:.0f}%                        │")
            print (f"│  4. Model path              : {str(model_path or 'default'):<35} │")
            print (f"└{'─'*68}┘")
            sub =input ("   Ubah nomor mana? (Enter = batal): ").strip ()
            if sub =="1":
                val =input (f"   Top-K [{top_k}]: ").strip ()
                if val .isdigit ():top_k =int (val )
            elif sub =="2":
                val =input (f"   Kandidat [{n_candidates}]: ").strip ()
                if val .isdigit ():n_candidates =int (val )
            elif sub =="3":
                val =input (f"   Min melody score (0-100) [{min_melody*100:.0f}]: ").strip ()
                try :
                    f =float (val )
                    min_melody =f /100.0 if f >1 else f 
                except ValueError :
                    pass 
            elif sub =="4":
                val =input ("   Model path (Enter = default): ").strip ()
                model_path =val if val else None 
            print ("   ✅ Pengaturan disimpan.")
            continue 
        input_path =raw 
        if not is_youtube_url (input_path )and not os .path .exists (input_path ):
            print (f"   ❌ File tidak ditemukan: {input_path}")
            print ("      Coba lagi atau ketik URL YouTube.")
            continue 
        print (f"\n   ⚙️  top-k={top_k}  kandidat={n_candidates}  "
        f"min-melody={min_melody*100:.0f}%  "
        f"(ketik 's' untuk ubah)")
        try :
            melody_first_query (
            input_path ,
            top_k =top_k ,
            n_candidates =n_candidates ,
            model_path =model_path ,
            min_melody_score =min_melody ,
            )
        except KeyboardInterrupt :
            print ("\n\n⚠️  Query dibatalkan. Ketik 'q' untuk keluar.")
        print ("\n"+"─"*W )
        print ("  Query berikutnya? (atau 'q' untuk keluar, 's' untuk settings)")
def main ():
    parser =argparse .ArgumentParser (
    description =textwrap .dedent ("""\
            🎼 Melody-First Similarity Search
            Jalankan TANPA argumen untuk mode interaktif (loop).
            Jalankan DENGAN argumen untuk mode satu-query CLI.
            Contoh:
              python script/melody_first.py                          # interaktif
              python script/melody_first.py audio.mp3               # satu file
              python script/melody_first.py "https://youtu.be/..."  # YouTube
        """),
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    )
    parser .add_argument ("input",nargs ="?",default =None ,
    help ="File audio query (.wav .mp3 dll) atau YouTube URL. "
    "Kosongkan untuk mode interaktif.")
    parser .add_argument ("--top-k",type =int ,default =5 ,
    help ="Jumlah hasil per query segment (default: 5)")
    parser .add_argument ("--candidates",type =int ,default =80 ,
    help ="Jumlah kandidat embedding pre-filter (default: 80)")
    parser .add_argument ("--min-melody",type =float ,default =0.60 ,
    help ="Minimum melody score [0.0-1.0] (default: 0.60)")
    parser .add_argument ("--model",type =str ,default =None ,
    help ="Override path model checkpoint")
    args =parser .parse_args ()
    if args .input is None :
        _interactive_mode ()
    else :
        if not is_youtube_url (args .input )and not os .path .exists (args .input ):
            print (f"❌ File tidak ditemukan: {args.input}")
            sys .exit (1 )
        melody_first_query (
        args .input ,
        top_k =args .top_k ,
        n_candidates =args .candidates ,
        model_path =args .model ,
        min_melody_score =args .min_melody ,
        )
if __name__ =="__main__":
    main ()

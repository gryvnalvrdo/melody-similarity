import time as _time
import os 
import gc 
import signal 
os .environ ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"]="1"
import numpy as np 
import librosa 
from tqdm import tqdm 
def init_worker ():
    os .environ ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"]="1"
    signal .signal (signal .SIGINT ,signal .SIG_IGN )
import argparse 
from pathlib import Path 
                                                                                 
try :
    from demucs_utils import remove_drums as _remove_drums 
    _DEMUCS_AVAILABLE =True 
except ImportError :
    try :
        from script .demucs_utils import remove_drums as _remove_drums 
        _DEMUCS_AVAILABLE =True 
    except ImportError :
        _remove_drums =None 
        _DEMUCS_AVAILABLE =False 
        print ("[extract_features] ⚠️  demucs_utils tidak ditemukan — menggunakan raw audio")
try :
    from config import (
    DATASET_DIR ,FEATURES_DIR ,SAMPLE_RATE ,
    SEGMENT_DURATION ,SEGMENT_HOP ,MIN_SEGMENT_DURATION ,
    N_CQT_BINS ,HOP_LENGTH ,BINS_PER_OCTAVE ,MIN_NOTE ,
    FRAMES_PER_SEGMENT 
    )
    from music_theory import get_key_from_cqt ,get_chromatic_index 
except ImportError :
    from script .config import (
    DATASET_DIR ,FEATURES_DIR ,SAMPLE_RATE ,
    SEGMENT_DURATION ,SEGMENT_HOP ,MIN_SEGMENT_DURATION ,
    N_CQT_BINS ,HOP_LENGTH ,BINS_PER_OCTAVE ,MIN_NOTE ,
    FRAMES_PER_SEGMENT 
    )
    from script .music_theory import get_key_from_cqt ,get_chromatic_index 
def load_audio (file_path ,sr =SAMPLE_RATE ):
    import torchaudio 
    try :
        audio ,orig_sr =torchaudio .load (file_path )
        if audio .shape [0 ]>1 :
            audio =audio .mean (dim =0 ,keepdim =True )
        max_frames =int (360 *orig_sr )
        if audio .shape [1 ]>max_frames :
            audio =audio [:,:max_frames ]
        if orig_sr !=sr :
            resampler =torchaudio .transforms .Resample (orig_sr ,sr )
            audio =resampler (audio )
        return audio .squeeze (0 ).numpy ()
    except Exception as e :
        import warnings 
        try :
            with warnings .catch_warnings ():
                warnings .simplefilter ("ignore")
                import librosa 
                audio ,_ =librosa .load (file_path ,sr =sr ,mono =True ,res_type ='kaiser_fast',duration =360.0 )
            return audio 
        except Exception as e2 :
            print (f"Error loading {file_path}: {e} / {e2}")
            return None 
def _apply_drum_removal (audio ):
    if not _DEMUCS_AVAILABLE or _remove_drums is None :
        return audio 
    result =_remove_drums (audio ,SAMPLE_RATE )
    if result is None :
        return audio            
    return result 

def _cqt_from_clean_audio(audio):
    audio_no_drums = _apply_drum_removal(audio)
    cqt = np.abs(librosa.cqt(audio_no_drums, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_bins=N_CQT_BINS, bins_per_octave=BINS_PER_OCTAVE))
    cqt_norm = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt_norm = (cqt_norm - cqt_norm.min()) / (cqt_norm.max() - cqt_norm.min() + 1e-6)
    return cqt_norm, cqt
def extract_full_cqt (audio ,verbose =True ):
    t0 =_time .time ()
    if verbose :
        dur_s =len (audio )/SAMPLE_RATE
        demucs_str ="Demucs (bass+other+vocals)"if _DEMUCS_AVAILABLE else "Raw Audio (Demucs N/A)"
        print (f"   🎵  [1/2] Drum Separation via {demucs_str} "
        f"(audio: {dur_s:.0f}s)...")
        print (f"   📊  [2/2] CQT computation (no drums)...")
        
    result =_cqt_from_clean_audio (audio )
    
    if verbose :
        n_bins ,n_frames =result [0 ].shape
        print (f"   ✅  CQT done  ({n_bins} bins × {n_frames} frames) in {_time.time()-t0:.1f}s")
        
    return result
def get_features_for_audio (audio ,verbose =True ):
    t0 =_time .time ()
    try :
        cqt_norm ,cqt_mag =extract_full_cqt (audio ,verbose =verbose )
    except Exception as e :
        print (f"Error extracting CQT: {e}")
        return None 
    if verbose :
        print (f"   🔑  [3/3] Key detection...")
    global_key ,confidence =get_key_from_cqt (cqt_mag )

    _root_note =global_key .split (' ')[0 ]
    _mode      =global_key .split (' ')[1 ]if ' 'in global_key else 'Major'
    _root_idx  =get_chromatic_index (_root_note )

    _KEY_CONFIDENCE_THRESHOLD =0.15
    is_minor = ('Minor' in global_key and 'low-conf' not in global_key)
    if confidence < _KEY_CONFIDENCE_THRESHOLD:
        global_key = global_key + " (low-conf)"
        shift_to_c = 0
        _effective_mode = 'unknown'
    elif is_minor:
        shift_to_c = (9 - _root_idx) % 12
        if shift_to_c > 6:
            shift_to_c -= 12
        _effective_mode = 'minor'
    else:
        shift_to_c = -_root_idx if _root_idx != -1 else 0
        _effective_mode = 'major'

    _bins_per_semitone =BINS_PER_OCTAVE //12
    cqt_shifted =np .roll (cqt_norm ,shift_to_c *_bins_per_semitone ,axis =0 )
    n_frames =cqt_shifted .shape [1 ]
    hop_frames =int (SEGMENT_HOP *SAMPLE_RATE /HOP_LENGTH )
    seg_frames =int (SEGMENT_DURATION *SAMPLE_RATE /HOP_LENGTH )
    min_seg_frames =int (MIN_SEGMENT_DURATION *SAMPLE_RATE /HOP_LENGTH )
    segments =[]
    metadata =[]
    current_frame =0
    while current_frame +min_seg_frames <=n_frames :
        start_frame =current_frame
        end_frame =min (current_frame +seg_frames ,n_frames )
        seg =cqt_shifted [:,start_frame :end_frame ]
        if seg .shape [1 ]<seg_frames :
            pad_width =seg_frames -seg .shape [1 ]
            seg =np .pad (seg ,((0 ,0 ),(0 ,pad_width )),mode ='constant')
        segments .append (seg )
        _seg_start_time = start_frame * HOP_LENGTH / SAMPLE_RATE
        metadata .append ({
        'start_frame':start_frame ,
        'end_frame':end_frame ,
        'start_time': _seg_start_time,
        'end_time':   _seg_start_time + SEGMENT_DURATION,
        'key_confidence':float (confidence ),
        'shift_to_c':int (shift_to_c ),
        'mode': _effective_mode,
        'global_key': global_key,
        })
        current_frame +=hop_frames
    if not segments :
        return None
    return {
    'harmonic_cqt':np .array (segments ,dtype =np .float32 ),
    'metadata':metadata ,
    'global_key':global_key ,
    'shift_to_c':int (shift_to_c ),
    'key_confidence':float (confidence ),
    'mode': _effective_mode,
    }

    orig_audio =load_audio (orig_input )if isinstance (orig_input ,(str ,Path ))else orig_input 
    cover_audio =load_audio (cover_input )if isinstance (cover_input ,(str ,Path ))else cover_input 
    if orig_audio is None or cover_audio is None :
        return None ,{'global_similarity':0.0 }
    try :
        orig_cqt ,orig_mag =extract_full_cqt (orig_audio )
        cover_cqt ,cover_mag =extract_full_cqt (cover_audio )
    except Exception as e :
        print (f"Error extracting CQT: {e}")
        return None ,{'global_similarity':0.0 }
    global_key :str ="C major"
    confidence :float =0.0 
    shift_to_c :int =0 
    global_similarity :float =0.0 
    aligned_cover_cqt =cover_cqt 
    _KEY_CONFIDENCE_THRESHOLD =0.15 
    try :
        eps =1e-8 
        orig_chroma =librosa .feature .chroma_cqt (C =orig_mag ,sr =SAMPLE_RATE ,hop_length =HOP_LENGTH ,bins_per_octave =BINS_PER_OCTAVE )
        cover_chroma =librosa .feature .chroma_cqt (C =cover_mag ,sr =SAMPLE_RATE ,hop_length =HOP_LENGTH ,bins_per_octave =BINS_PER_OCTAVE )
        global_key ,confidence =get_key_from_cqt (orig_mag )
        _root_note =global_key .split (' ')[0 ]
        _root_idx =get_chromatic_index (_root_note )
        if confidence >=_KEY_CONFIDENCE_THRESHOLD and _root_idx !=-1 :
            shift_to_c =-_root_idx 
        else :
            shift_to_c =0 
            global_key =global_key +" (low-conf, unshifted)"
        del orig_mag ,cover_mag 
        downsample_factor =16 
        orig_chroma_ds =orig_chroma [:,::downsample_factor ]
        cover_chroma_ds =cover_chroma [:,::downsample_factor ]
        del orig_chroma ,cover_chroma 
        gc .collect ()
        orig_profile =np .sum (orig_chroma_ds ,axis =1 )
        cover_profile =np .sum (cover_chroma_ds ,axis =1 )
        orig_profile_norm =orig_profile /(np .linalg .norm (orig_profile )+eps )
        cover_profile_norm =cover_profile /(np .linalg .norm (cover_profile )+eps )
        best_shift =0 
        best_corr =-1 
        for shift in range (12 ):
            shifted_cover =np .roll (cover_profile_norm ,shift )
            corr =np .dot (orig_profile_norm ,shifted_cover )
            if corr >best_corr :
                best_corr =corr 
                best_shift =shift 
        if best_shift !=0 :
            cover_chroma_ds =np .roll (cover_chroma_ds ,best_shift ,axis =0 )
            bins_per_semitone =BINS_PER_OCTAVE //12 
            cover_cqt =np .roll (cover_cqt ,best_shift *bins_per_semitone ,axis =0 )
        safe_orig =orig_chroma_ds .astype (np .float32 )+eps 
        safe_cover =cover_chroma_ds .astype (np .float32 )+eps 
        del orig_chroma_ds ,cover_chroma_ds 
        D ,wp_ds =librosa .sequence .dtw (X =safe_orig ,Y =safe_cover ,metric ='cosine')
        del safe_orig ,safe_cover 
        path_length =len (wp_ds )
        avg_distance =D [-1 ,-1 ]/path_length if path_length >0 else 2.0 
        global_similarity =max (0.0 ,1.0 -(avg_distance /2.0 ))
        wp_ds =wp_ds [::-1 ]
        del D 
        path_lookup :dict ={}
        for ds_o ,ds_c in wp_ds :
            if ds_o not in path_lookup :
                path_lookup [ds_o ]=[]
            path_lookup [ds_o ].append (int (ds_c ))
        path_map ={k :int (np .median (v ))for k ,v in path_lookup .items ()}
        del wp_ds ,path_lookup 
        aligned_cover_cqt =np .zeros_like (orig_cqt )
        last_cover_idx =0 
        for orig_idx in range (orig_cqt .shape [1 ]):
            ds_orig_idx =orig_idx //downsample_factor 
            if ds_orig_idx in path_map :
                last_cover_idx =min (
                path_map [ds_orig_idx ]*downsample_factor ,
                cover_cqt .shape [1 ]-1 
                )
            aligned_cover_cqt [:,orig_idx ]=cover_cqt [:,last_cover_idx ]
    except Exception as e :
        print (f"DTW Error: {e} — using unaligned fallback")
        min_frames =min (orig_cqt .shape [1 ],cover_cqt .shape [1 ])
        orig_cqt =orig_cqt [:,:min_frames ]
        aligned_cover_cqt =cover_cqt [:,:min_frames ]
    return (orig_cqt ,aligned_cover_cqt ,global_key ,confidence ,shift_to_c ),{'global_similarity':global_similarity }
def _process_song_folder (args ):
    song_dir ,output_song_dir =args 
    orig_path =os .path .join (song_dir ,"original.wav")
    song_id =os .path .basename (song_dir )
    if not os .path .exists (orig_path ):
        return {'status':'failed','song_id':song_id ,'song_dir':song_dir ,'reason':'File original.wav tidak ditemukan'}
    orig_audio =load_audio (orig_path )
    if orig_audio is None :
        import shutil ;shutil .rmtree (output_song_dir ,ignore_errors =True )
        return {'status':'failed','song_id':song_id ,'song_dir':song_dir ,'reason':'Gagal memuat original audio'}
    cover_files =['cover','cover_2','cover_3']
    processed_any_cover =False 

    orig_npy_path =os .path .join (output_song_dir ,"original.npy")
    if not os .path .exists (orig_npy_path ):
        demucs_tag ="Demucs drum-removed CQT"if _DEMUCS_AVAILABLE else "raw CQT (Demucs N/A)"
        print (f"   🔄 Processing original ({demucs_tag}, no DTW)...")
        orig_features =get_features_for_audio (orig_audio ,verbose =True )
        if orig_features is None :
            import shutil ;shutil .rmtree (output_song_dir ,ignore_errors =True )
            return {'status':'failed','song_id':song_id ,'song_dir':song_dir ,'reason':'Gagal extract fitur original'}
        orig_features ['source_file']=str (orig_path )
        orig_features ['demucs_applied']=_DEMUCS_AVAILABLE 
        np .save (orig_npy_path ,orig_features ,allow_pickle =True )
        del orig_features 
        gc .collect ()
    del orig_audio 
    gc .collect ()

    for cover_name in cover_files :
        cover_path =os .path .join (song_dir ,f"{cover_name}.wav")
        if not os .path .exists (cover_path ):
            continue 
        cover_audio =load_audio (cover_path )
        if cover_audio is None :
            continue 
        demucs_tag ="Demucs drum-removed CQT"if _DEMUCS_AVAILABLE else "raw CQT (Demucs N/A)"
        print (f"   🔄 Processing {cover_name} ({demucs_tag}, no DTW)...")

        cover_features =get_features_for_audio (cover_audio ,verbose =True )
        if cover_features is None :
            del cover_audio 
            gc .collect ()
            continue 
        cover_features ['source_file']=str (cover_path )
        cover_features ['demucs_applied']=_DEMUCS_AVAILABLE 
        np .save (os .path .join (output_song_dir ,f"{cover_name}.npy"),cover_features ,allow_pickle =True )
        processed_any_cover =True 
        del cover_audio ,cover_features 
        gc .collect ()
    if not processed_any_cover :
        import shutil ;shutil .rmtree (output_song_dir ,ignore_errors =True )
        return {'status':'failed','song_id':song_id ,'song_dir':song_dir ,'reason':'Tidak ada cover yang valid/berhasil di-extract'}
    gc .collect ()
    import shutil 
    shutil .rmtree (song_dir ,ignore_errors =True )
    gc .collect ()
    try :
        import torch 
        if torch .cuda .is_available ():
            torch .cuda .empty_cache ()
    except Exception :
        pass 
    return {'status':'success','song_id':song_id }
def process_dataset (dataset_dir =DATASET_DIR ,output_dir =FEATURES_DIR ,
force =False ,workers =1 ,
start_song =None ,end_song =None ,
reraise_interrupt =False ):
    def _norm (s ):
        try :
            return str (int (s )).zfill (5 )
        except Exception :
            return s 
    start_norm =_norm (start_song )if start_song else None 
    end_norm =_norm (end_song )if end_song else None 
    all_folders =sorted ([
    f for f in os .listdir (dataset_dir )
    if os .path .isdir (os .path .join (dataset_dir ,f ))
    ],key =lambda x :(0 ,int (x ))if x .isdigit ()else (1 ,x ))
    song_folders =[]
    for f in all_folders :
        f_norm =_norm (f )
        if start_norm and f_norm <start_norm :
            continue 
        if end_norm and f_norm >end_norm :
            continue 
        song_folders .append (f )
    tasks =[]
    skipped_count =0 
    for song_id in song_folders :
        song_dir =os .path .join (dataset_dir ,song_id )
        output_song_dir =os .path .join (output_dir ,song_id )
        orig_wav =os .path .join (song_dir ,"original.wav")
        has_any_cover =any (os .path .exists (os .path .join (song_dir ,f"{c}.wav"))for c in ["cover","cover_2","cover_3"])
        if not (os .path .exists (orig_wav )and has_any_cover ):
            continue 
        try :
            os .rename (orig_wav ,orig_wav )
        except OSError :
            continue 
        orig_ny =os .path .join (output_song_dir ,"original.npy")
        cover_ny =os .path .join (output_song_dir ,"cover.npy")
        all_extracted =True 
        if not os .path .exists (orig_ny )or not os .path .exists (cover_ny ):
            all_extracted =False 
        else :
            for cv in ['cover_2','cover_3']:
                wav_path =os .path .join (song_dir ,f"{cv}.wav")
                npy_path =os .path .join (output_song_dir ,f"{cv}.npy")
                if os .path .exists (wav_path )and not os .path .exists (npy_path ):
                    all_extracted =False 
                    break 
        if all_extracted and not force :
            skipped_count +=1 
            import shutil 
            shutil .rmtree (song_dir ,ignore_errors =True )
            continue 
        os .makedirs (output_song_dir ,exist_ok =True )
        tasks .append ((song_dir ,output_song_dir ))
    import multiprocessing 
    try :
        import torch 
        has_cuda =torch .cuda .is_available ()
    except Exception :
        has_cuda =False 
    if workers is None or workers <=0 :
        if has_cuda :
            try :
                import torch 
                total_vram =torch .cuda .get_device_properties (0 ).total_memory 
                n_workers =max (1 ,int (total_vram /(3.0 *1024 **3 )))
                n_workers =min (n_workers ,max (1 ,multiprocessing .cpu_count ()//2 ))
                worker_mode_str =f"🚀 CUDA + AMP. Auto-assigned {n_workers} workers (VRAM: {total_vram//1024**3}GB)."
            except Exception :
                n_workers =1 
                worker_mode_str ="🚀 CUDA detected. Auto-assigned 1 worker (fallback)."
        else :
            n_workers =max (1 ,multiprocessing .cpu_count ()//2 )
            worker_mode_str =f"🖥️ CPU mode. Auto-assigned {n_workers} workers."
    else :
        n_workers =workers 
        worker_mode_str =f"🚀 Using {n_workers} user-requested workers."
    if not tasks :
        return 0 
    demucs_banner ="Demucs Drum-Removed"if _DEMUCS_AVAILABLE else "Raw Audio (Demucs N/A)"
    print ("="*70 )
    print (f"\U0001f3b5 CQT FEATURE EXTRACTION [{demucs_banner}] (bass+other+vocals, independent per-version)")
    print ("="*70 )
    if start_song or end_song :
        print (f"  Range: {start_song or 'first'} .. {end_song or 'last'}")
    print (f"\n📊 Found {len(song_folders)} song folders")
    print (f"📊 Folders to process: {len(tasks)} (skipped: {skipped_count})")
    print (worker_mode_str +"\n")
    success_count =0 
    failed_count =0 
    failed_songs =[]
    ctx =multiprocessing .get_context ('spawn')
    pool =ctx .Pool (processes =n_workers ,initializer =init_worker )
    try :
        results =pool .imap_unordered (_process_song_folder ,tasks )
        for result in tqdm (results ,total =len (tasks ),desc ="Extracting (Raw Audio CQT)"):
            if result ['status']=='success':
                success_count +=1 
            else :
                failed_count +=1 
                failed_songs .append (result )
                tqdm .write (f"⚠️ GAGAL [Folder {result['song_id']}]: {result.get('reason', 'Kesalahan tidak diketahui')}")
                import shutil 
                shutil .rmtree (result ['song_dir'],ignore_errors =True )
        pool .close ()
        pool .join ()
    except KeyboardInterrupt :
        print ("\n\n🛑 Ctrl+C received — cleanly stopping workers...")
        pool .terminate ()
        pool .join ()
        print ("🧹 Sweeping incomplete feature folders...")
        import shutil as _shutil 
        _swept =[]
        for _sid in os .listdir (output_dir ):
            _sid_dir =os .path .join (output_dir ,_sid )
            if not os .path .isdir (_sid_dir ):
                continue 
            _has_orig =os .path .exists (os .path .join (_sid_dir ,'original.npy'))
            _has_cover =os .path .exists (os .path .join (_sid_dir ,'cover.npy'))
            if not (_has_orig and _has_cover ):
                _shutil .rmtree (_sid_dir ,ignore_errors =True )
                _swept .append (_sid )
        if _swept :
            print (f"   Removed {len(_swept)} incomplete: "
            f"{', '.join(_swept[:10])}{'...' if len(_swept) > 10 else ''}")
        else :
            print ("   Nothing to sweep \u2014 all feature folders are clean.")
        print ("\u2705 Feature directory is clean. Resume with the same command.")
        if reraise_interrupt :
            raise 
    print ("\n"+"="*70 )
    print ("📊 EXTRACTION SUMMARY")
    print ("="*70 )
    demucs_note ="Demucs drum-removed"if _DEMUCS_AVAILABLE else "raw audio (Demucs N/A)"
    print (f"\u2705 Success: {success_count} songs extracted ({demucs_note} CQT)")
    print (f"⏭️  Skipped: {skipped_count}")
    print (f"❌ Failed/Filtered: {failed_count}")
    if failed_songs :
        print ("\n🧹 Initiating Auto-Cleanup for failed songs...")
        import shutil 
        import csv 
        failed_ids =set ([r ['song_id']for r in failed_songs ])
        for f_song in failed_songs :
            shutil .rmtree (f_song ['song_dir'],ignore_errors =True )
            shutil .rmtree (os .path .join (output_dir ,f_song ['song_id']),ignore_errors =True )
        try :
            from config import EXHAUSTED_CSV_PATH 
        except ImportError :
            from script .config import EXHAUSTED_CSV_PATH 
        print (f"   -> Skipped deleting {len(failed_ids)} bad entries from dataset.csv (User Override)")
        if os .path .exists (EXHAUSTED_CSV_PATH ):
            with open (EXHAUSTED_CSV_PATH ,"a",newline ="",encoding ="utf-8")as f :
                writer =csv .writer (f )
                for fid in failed_ids :
                    writer .writerow (["Unknown Artist",f"Song {fid}","Failed strict DTW alignment extraction"])
    print ("="*70 )
    return success_count 
if __name__ =="__main__":
    parser =argparse .ArgumentParser (
    description ="Extract Demucs-CQT features from audio pairs."
    )
    parser .add_argument ("--test",action ="store_true",help ="Test on one song pair")
    parser .add_argument ("--force",action ="store_true",help ="Re-extract even if .npy exists")
    parser .add_argument ("--workers",type =int ,default =0 ,
    help ="Parallel workers (0=auto; Demucs is GPU-bound, more workers = more GPU memory)")
    parser .add_argument ("--start-song",type =str ,default =None ,
    help ="Process songs with ID >= this value (e.g. '001' or '4001')")
    parser .add_argument ("--end-song",type =str ,default =None ,
    help ="Process songs with ID <= this value (e.g. '4000' or '5000')")
    parser .add_argument ("--watch",action ="store_true",
    help ="Watch mode: keep running and process new downloads automatically")
    parser .add_argument ("--interval",type =int ,default =30 ,
    help ="Seconds between scans in watch mode (default: 30)")
    args =parser .parse_args ()
    if args .test :
        print ("\n TESTING RAW CQT EXTRACTION (Pair #1)\n")
        test_dir =None 
        for song_id in sorted (os .listdir (DATASET_DIR )):
            song_dir =os .path .join (DATASET_DIR ,song_id )
            if (os .path .isdir (song_dir )
            and os .path .exists (os .path .join (song_dir ,"original.wav"))
            and os .path .exists (os .path .join (song_dir ,"cover.wav"))):
                test_dir =song_dir 
                break 
        if test_dir is None :
            print ("❌ No valid audio pairs found for testing")
        else :
            print (f"📂 Test folder: {test_dir}")
            res =_process_song_folder ((test_dir ,os .path .join (FEATURES_DIR ,os .path .basename (test_dir ))))
            print (f"\n✅ Test Result: {res}")
    elif args .watch :
        import time 
        print ("\n"+"="*70 )
        print ("👁️  WATCH MODE — extracting as songs are downloaded")
        print ("   Will scan for new songs every",args .interval ,"seconds")
        print ("   Press Ctrl+C to stop")
        print ("="*70 +"\n")
        while True :
            try :
                n =process_dataset (
                force =args .force ,
                workers =args .workers ,
                start_song =args .start_song ,
                end_song =args .end_song ,
                reraise_interrupt =True ,
                )
                if n ==0 :
                    import sys 
                    sys .stdout .write (f"\r⏳ [Watch] Scanning... No newly completed downloads. Next check in {args.interval}s")
                    sys .stdout .flush ()
                    time .sleep (args .interval )
                else :
                    print (f"\n✅ Processed {n} song pair(s). Scanning for more...\n")
            except KeyboardInterrupt :
                print ("\n✅ Watch mode stopped.")
                break 
    else :
        process_dataset (
        force =args .force ,
        workers =args .workers ,
        start_song =args .start_song ,
        end_song =args .end_song ,
        )

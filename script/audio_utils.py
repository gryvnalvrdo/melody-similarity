import os 
import tempfile 
import numpy as np 
try :
    from config import SAMPLE_RATE 
except ImportError :
    from script .config import SAMPLE_RATE 
def normalize_query_audio (input_path :str )->str :
    import librosa 
    import soundfile as sf 
    try :
        audio ,orig_sr =librosa .load (input_path ,sr =None ,mono =True ,res_type ='kaiser_fast')
    except Exception as e :
        print (f"   ⚠️  audio_utils: librosa failed to load '{input_path}': {e}")
        print ("       Falling back to original path (torchaudio may handle it).")
        return input_path 
    if orig_sr !=SAMPLE_RATE :
        audio =librosa .resample (audio ,orig_sr =orig_sr ,target_sr =SAMPLE_RATE )
    peak =np .max (np .abs (audio ))
    if peak >1e-6 :
        target_peak =10 **(-3.0 /20.0 )
        audio =audio *(target_peak /peak )
    tmp_dir =os .path .join (
    os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),
    "downloads","normalized"
    )
    os .makedirs (tmp_dir ,exist_ok =True )
    base_name =os .path .splitext (os .path .basename (input_path ))[0 ]
    fd ,tmp_path =tempfile .mkstemp (suffix =".wav",prefix =f"{base_name}_norm_",dir =tmp_dir )
    os .close (fd )
    sf .write (tmp_path ,audio ,SAMPLE_RATE ,subtype ="PCM_16")
    print (f"   🎛️  Normalized audio → {os.path.basename(tmp_path)}"
    f"  ({orig_sr} Hz → {SAMPLE_RATE} Hz, peak → -3 dBFS)")
    return tmp_path 
def cleanup_normalized_audio (tmp_path :str ,original_path :str )->None :
    if tmp_path !=original_path and os .path .exists (tmp_path ):
        try :
            os .remove (tmp_path )
        except Exception :
            pass 
def extract_melody_segment (audio_path :str ,start_time :float ,end_time :float ,
output_path :str )->bool :
    import soundfile as sf 
    if not os .path .exists (audio_path ):
        return False 
    duration =max (end_time -start_time ,1.0 )
    try :
        import librosa 
        audio ,_ =librosa .load (
        audio_path ,sr =SAMPLE_RATE ,mono =True ,
        offset =start_time ,duration =duration +1.0 ,
        res_type ='kaiser_fast'
        )
    except Exception as e :
        print (f"   ⚠️  audio_utils: cannot load '{audio_path}': {e}")
        return False 
    melody =audio 
    target_n =int (duration *SAMPLE_RATE )
    if len (melody )>target_n :
        melody =melody [:target_n ]
    peak =np .max (np .abs (melody ))
    if peak >1e-6 :
        melody =melody *(0.708 /peak )
    melody_i16 =(melody *32767 ).astype (np .int16 )
    try :
        out_dir =os .path .dirname (output_path )
        if out_dir :
            os .makedirs (out_dir ,exist_ok =True )
        sf .write (output_path ,melody_i16 ,SAMPLE_RATE )
        return True 
    except Exception as e :
        print (f"   ⚠️  audio_utils: cannot write '{output_path}': {e}")
        return False 
def redetect_notes_from_audio (wav_path :str ,stored_notes :np .ndarray ,
tolerance :int =1 )->tuple :
    if not os .path .exists (wav_path )or len (stored_notes )==0 :
        return None ,0 ,len (stored_notes ),[]
    try :
        import librosa 
    except ImportError :
        return None ,0 ,len (stored_notes ),[]
    try :
        audio ,_ =librosa .load (wav_path ,sr =SAMPLE_RATE ,mono =True )
        f0 =librosa .yin (
        audio ,
        fmin =librosa .note_to_hz ('C2'),
        fmax =librosa .note_to_hz ('C7'),
        sr =SAMPLE_RATE ,
        hop_length =512 ,
        )
        voiced =f0 [f0 >50.0 ]
        if len (voiced )==0 :
            return None ,0 ,len (stored_notes ),[]
        midi_cont =12.0 *np .log2 (voiced /440.0 )+69.0 
        midi_q =np .round (midi_cont ).astype (int )
        min_frames =4 
        redetected :list =[]
        run_note =-999 
        run_count =0 
        for n in midi_q :
            valid =21 <=n <=108 
            if valid and n ==run_note :
                run_count +=1 
            else :
                if run_count >=min_frames and run_note !=-999 :
                    redetected .append (int (run_note ))
                run_note =n if valid else -999 
                run_count =1 if valid else 0 
        if run_count >=min_frames and run_note !=-999 :
            redetected .append (int (run_note ))
        stored_list =[int (n )for n in stored_notes ]
        used :set =set ()
        matches =0 
        for s in stored_list :
            for i ,r in enumerate (redetected ):
                if i not in used and abs (s -r )<=tolerance :
                    matches +=1 
                    used .add (i )
                    break 
        total =len (stored_list )
        pct =(matches /total *100.0 )if total >0 else 0.0 
        return pct ,matches ,total ,redetected 
    except Exception as e :
        print (f"   ⚠️  audio_utils: re-detection failed on '{wav_path}': {e}")
        return None ,0 ,len (stored_notes ),[]
def prefetch_proof_songs (song_tasks :list ,session_cache :dict ,
max_workers :int =4 )->None :
    import concurrent .futures 
    global _CSV_URL_CACHE 
    if not _CSV_URL_CACHE :
        _CSV_URL_CACHE =_load_csv_urls ()
    try :
        import yt_dlp 
    except ImportError :
        print ("   ⚠️  audio_utils: yt-dlp not installed — skipping parallel prefetch")
        return 
    unique_tasks =list ({(_normalize_song_id (str (s )),str (v ).lower ())
    for s ,v in song_tasks 
    if (_normalize_song_id (str (s )),str (v ).lower ())not in session_cache })
    if not unique_tasks :
        return 
    print (f"   ⏬ Pre-fetching {len(unique_tasks)} song(s) in parallel "
    f"(max {max_workers} workers)…")
    def _download_one (sid_norm_ver :tuple )->None :
        sid_norm ,ver =sid_norm_ver 
        url =_CSV_URL_CACHE .get ((sid_norm ,ver ),'')
        if not url :
            print (f"   ⚠️  No URL for song {sid_norm} ({ver}) — skipping")
            return 
        import tempfile ,os 
        tmp_dir =tempfile .mkdtemp (prefix ="proof_dl_")
        base =os .path .join (tmp_dir ,"audio")
        ydl_opts ={
        'format':'bestaudio/best',
        'postprocessors':[{'key':'FFmpegExtractAudio',
        'preferredcodec':'wav','preferredquality':'0'}],
        'outtmpl':f'{base}.%(ext)s',
        'quiet':True ,
        'no_warnings':True ,
        'retries':3 ,
        'socket_timeout':60 ,
        'nocheckcertificate':True ,
        'extractor_args':{'youtube':{'player_client':['android','ios','web']}},
        }
        import yt_dlp as _yt 
        try :
            with _yt .YoutubeDL (ydl_opts )as ydl :
                ydl .download ([url ])
            for ext in ['wav','mp3','m4a','webm','opus']:
                cand =f"{base}.{ext}"
                if os .path .exists (cand ):
                    session_cache [(sid_norm ,ver )]=cand 
                    print (f"   ✅ Downloaded song {sid_norm} ({ver})")
                    return 
            print (f"   ⚠️  Downloaded but file not found for song {sid_norm}")
        except Exception as e :
            print (f"   ⚠️  Prefetch failed for song {sid_norm}: {e}")
        finally :
            if (sid_norm ,ver )not in session_cache :
                import shutil 
                shutil .rmtree (tmp_dir ,ignore_errors =True )
    with concurrent .futures .ThreadPoolExecutor (max_workers =max_workers )as pool :
        list (pool .map (_download_one ,unique_tasks ))
    cached =sum (1 for k in unique_tasks if k in session_cache )
    print (f"   ✅ Prefetch complete: {cached}/{len(unique_tasks)} songs ready in cache")
def _normalize_song_id (sid :str )->str :
    try :
        return str (int (sid ))
    except (ValueError ,TypeError ):
        return str (sid ).strip ()
def _load_csv_urls ()->dict :
    try :
        try :
            from config import DATASET_CSV_PATH 
        except ImportError :
            from script .config import DATASET_CSV_PATH 
        import csv 
        url_map ={}
        if not os .path .exists (DATASET_CSV_PATH ):
            return url_map 
        with open (DATASET_CSV_PATH ,encoding ='utf-8')as f :
            for row in csv .DictReader (f ):
                sid =_normalize_song_id (row .get ('song_id',''))
                ver =str (row .get ('version','')).strip ().lower ()
                url =str (row .get ('url','')).strip ()
                if sid and ver and url :
                    url_map [(sid ,ver )]=url 
        return url_map 
    except Exception :
        return {}
_CSV_URL_CACHE :dict ={}
def download_and_extract_proof_melody (song_id :str ,
version :str ,
start_time :float ,
end_time :float ,
output_path :str ,
session_cache :dict =None )->bool :
    global _CSV_URL_CACHE 
    if not _CSV_URL_CACHE :
        _CSV_URL_CACHE =_load_csv_urls ()
    sid_norm =_normalize_song_id (str (song_id ))
    url =_CSV_URL_CACHE .get ((sid_norm ,version .lower ()),'')
    if not url :
        print (f"   ⚠️  audio_utils: no URL found for song_id={song_id} (normalized={sid_norm}) version={version}")
        return False 
    try :
        import yt_dlp 
    except ImportError :
        print ("   ⚠️  audio_utils: yt-dlp not installed — cannot download proof audio")
        return False 
    cache_key =(sid_norm ,version .lower ())
    if session_cache is not None and cache_key in session_cache :
        cached_path =session_cache [cache_key ]
        if os .path .exists (cached_path ):
            print (f"   ♻️  Reusing cached audio for song {song_id} ({version})")
            return extract_melody_segment (cached_path ,start_time ,end_time ,output_path )
        else :
            del session_cache [cache_key ]
    import tempfile 
    tmp_dir =tempfile .mkdtemp (prefix ="proof_dl_")
    base =os .path .join (tmp_dir ,"audio")
    ydl_opts ={
    'format':'bestaudio/best',
    'postprocessors':[{'key':'FFmpegExtractAudio',
    'preferredcodec':'wav','preferredquality':'0'}],
    'outtmpl':f'{base}.%(ext)s',
    'quiet':True ,
    'no_warnings':True ,
    'retries':3 ,
    'socket_timeout':60 ,
    'nocheckcertificate':True ,
    'extractor_args':{'youtube':{'player_client':['android','ios','web']}},
    }
    wav_path =None 
    try :
        print (f"   ⏬ Downloading proof audio for song {song_id} ({version})…")
        with yt_dlp .YoutubeDL (ydl_opts )as ydl :
            ydl .download ([url ])
        for ext in ['wav','mp3','m4a','webm','opus']:
            cand =f"{base}.{ext}"
            if os .path .exists (cand ):
                wav_path =cand 
                break 
        if wav_path is None :
            print (f"   ⚠️  audio_utils: download completed but no audio file found")
            return False 
        if session_cache is not None :
            session_cache [cache_key ]=wav_path 
        else :
            ok =extract_melody_segment (wav_path ,start_time ,end_time ,output_path )
            import shutil ;shutil .rmtree (tmp_dir ,ignore_errors =True )
            return ok 
        return extract_melody_segment (wav_path ,start_time ,end_time ,output_path )
    except Exception as e :
        print (f"   ⚠️  audio_utils: download/extract failed for song {song_id}: {e}")
        import shutil ;shutil .rmtree (tmp_dir ,ignore_errors =True )
        return False 
def cleanup_proof_session_cache (session_cache :dict )->None :
    import shutil 
    cleaned =set ()
    for path in session_cache .values ():
        tmp_dir =os .path .dirname (path )
        if tmp_dir not in cleaned :
            shutil .rmtree (tmp_dir ,ignore_errors =True )
            cleaned .add (tmp_dir )
    session_cache .clear ()

import os 
import sys 
import numpy as np 
import torch 
import torch .nn .functional as F 
import librosa 
import argparse 
from tqdm import tqdm 
import re 
if sys .stdout .encoding !='utf-8':
    sys .stdout .reconfigure (encoding ='utf-8')
try :
    from config import (
    DEVICE ,SAMPLE_RATE ,SEGMENT_DURATION ,SEGMENT_HOP ,
    N_CQT_BINS ,HOP_LENGTH ,MODELS_DIR ,INDEX_DIR ,DATASET_DIR ,
    SIMILARITY_THRESHOLD ,TOP_K_RESULTS ,BINS_PER_OCTAVE ,
    MIN_CONSECUTIVE_MATCHES ,FEATURES_DIR 
    )
    from model import MelodySimilarityModel 
    from extract_features import load_audio ,get_features_for_audio 
    from build_index import load_model ,load_index 
    from music_theory import get_scale_notes 
except ImportError :
    from script .config import (
    DEVICE ,SAMPLE_RATE ,SEGMENT_DURATION ,SEGMENT_HOP ,
    N_CQT_BINS ,HOP_LENGTH ,MODELS_DIR ,INDEX_DIR ,DATASET_DIR ,
    SIMILARITY_THRESHOLD ,TOP_K_RESULTS ,BINS_PER_OCTAVE ,
    MIN_CONSECUTIVE_MATCHES ,FEATURES_DIR 
    )
    from script .model import MelodySimilarityModel 
    from script .extract_features import load_audio ,get_features_for_audio 
    from script .build_index import load_model ,load_index 
    from script .music_theory import get_scale_notes 
                                                                       
try :
    from contour_similarity import verify_candidates as _verify_fn 
except ImportError :
    try :
        from script .contour_similarity import verify_candidates as _verify_fn 
    except ImportError :
        _verify_fn =None 
verify_candidates =_verify_fn 
try :
    from audio_utils import normalize_query_audio ,cleanup_normalized_audio 
except ImportError :
    from script .audio_utils import normalize_query_audio ,cleanup_normalized_audio 

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
def extract_query_features (audio_path ):
    print (f"🎵 Loading audio: {audio_path}")
    audio =load_audio (audio_path )
    if audio is None :
        raise ValueError (f"Failed to load audio: {audio_path}")
    duration =len (audio )/SAMPLE_RATE 
    print (f"📊 Duration: {duration:.2f}s")
                                                                        
    result =get_features_for_audio (audio )
    if result is None :
        return None ,None 
    cqt_features =result .get ('harmonic_cqt')
    if cqt_features is None :
        cqt_features =result .get ('cqt_features')
    query_segments =result ['metadata']
    global_key =result ['global_key']
    for i ,seg in enumerate (query_segments ):
        feat =cqt_features [i ]
        seg ['features']=feat 
        seg ['nn_features']=feat 
        seg ['start_time']=seg .get ('start_time',i *SEGMENT_HOP )
    print (f"   ✅ Extracted {len(query_segments)} segments (drum-removed CQT). Key: {global_key}")
    return query_segments ,global_key 
def extract_query_embeddings (model ,query_segments ):
    model .eval ()
    embeddings =[]
    features_list =[s ['nn_features']if 'nn_features'in s else s ['features']for s in query_segments ]
    batch_size =32 
    dev = next(model.parameters()).device
    with torch .no_grad ():
        for i in range (0 ,len (features_list ),batch_size ):
            batch_data =features_list [i :i +batch_size ]
            batch_normalized =np .array (batch_data ,dtype =np .float32 )
            batch_tensor =torch .tensor (batch_normalized ).to (dev )
            emb =model (batch_tensor )
            embeddings .append (emb .cpu ().numpy ())
    if embeddings :
        return np .vstack (embeddings )
    return np .array ([])
def compute_cosine_similarity (query_emb ,index_embs ):
    q_tensor =torch .tensor (query_emb ,dtype =torch .float32 ).to (DEVICE )
    if q_tensor .dim ()==1 :
        q_tensor =q_tensor .unsqueeze (0 )
    batch_size =1024 
    all_sims =[]
    with torch .no_grad ():
        for i in range (0 ,len (index_embs ),batch_size ):
            batch =torch .tensor (index_embs [i :i +batch_size ],dtype =torch .float32 ).to (DEVICE )
            sims =F .cosine_similarity (q_tensor ,batch )
            all_sims .append (sims .cpu ().numpy ())
    return np .concatenate (all_sims )
def search (query_embedding ,index ,top_k =TOP_K_RESULTS ):
    if query_embedding .ndim ==1 :
        query_embedding =query_embedding [np .newaxis ,...]
    similarities =compute_cosine_similarity (query_embedding ,index ['embeddings'])
    top_indices =np .argsort (similarities )[::-1 ][:top_k ]
    results =[]
    for idx in top_indices :
        meta =index ['metadata'][idx ]
        results .append ({
        'song_id':meta ['song_id'],
        'version':meta ['version'],
        'segment_idx':meta ['segment_idx'],
        'start_time':meta ['start_time'],
        'end_time':meta ['end_time'],
        'similarity':float (similarities [idx ]),
        'global_key':meta .get ('global_key','Unknown')
        })
    return results 
def search_maxsim (all_embs ,index ,top_k =TOP_K_RESULTS , override_device=None):
    import sys
    cfg_device = override_device
    if cfg_device is None:
        if 'script.config' in sys.modules:
            cfg_device = sys.modules['script.config'].DEVICE
        elif 'config' in sys.modules:
            cfg_device = sys.modules['config'].DEVICE
        else:
            cfg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q_t =torch .tensor (all_embs ,dtype =torch .float32 ).to (cfg_device )
    q_t =F .normalize (q_t ,dim =-1 )
    idx_np =np .array (index ['embeddings'],dtype =np .float32 )
    norms =np .linalg .norm (idx_np ,axis =1 ,keepdims =True )
    norms [norms ==0 ]=1e-8 
    idx_np /=norms 
    idx_t =torch .from_numpy (idx_np ).to (cfg_device )
    meta =index ['metadata']
    song_ids =[m ['song_id']for m in meta ]
    unique_sids =list (dict .fromkeys (song_ids ))
    sid2idx ={s :i for i ,s in enumerate (unique_sids )}
    seg2song =torch .tensor (
    [sid2idx [m ['song_id']]for m in meta ],dtype =torch .long 
    ).to (cfg_device )
    N_q =q_t .shape [0 ]
    N_songs =len (unique_sids )
    sims =q_t @idx_t .T 
    per_q_max =torch .full ((N_q ,N_songs ),-2.0 ,device =cfg_device )
    idx_exp =seg2song .unsqueeze (0 ).expand (N_q ,-1 )
    per_q_max .scatter_reduce_ (1 ,idx_exp ,sims ,reduce ='amax',include_self =True )
                                                                
    mean_scores =per_q_max .mean (dim =0 )
    max_scores =per_q_max .max (dim =0 )[0 ]
    song_scores =0.40 *max_scores +0.60 *mean_scores 
    top_idxs =torch .argsort (song_scores ,descending =True )[:top_k ].cpu ().tolist ()
    sims_cpu =sims .cpu ().numpy ()
    seg2song_cpu =seg2song .cpu ().numpy ()
    results =[]
    for song_idx in top_idxs :
        sid =unique_sids [song_idx ]
        score =float (song_scores [song_idx ].item ())
        song_seg_indices =np .where (seg2song_cpu ==song_idx )[0 ]
        if len (song_seg_indices )==0 :
            continue 
        best_matches =[]
        for q_idx in range (N_q ):
            seg_sims =sims_cpu [q_idx ,song_seg_indices ]
            best_pos =int (np .argmax (seg_sims ))
            best_seg_idx =song_seg_indices [best_pos ]
            best_sim =float (seg_sims [best_pos ])
            seg_meta =meta [best_seg_idx ]
            best_matches .append ({
            'query_segment':q_idx ,
            'match_idx':seg_meta ['segment_idx'],
            'similarity':best_sim ,
            'start_time':seg_meta ['start_time'],
            'end_time':seg_meta ['end_time'],
            'version':seg_meta ['version'],
            'match_global_key':seg_meta .get ('global_key','Unknown')
            })
        versions =[m ['version']for m in best_matches ]
        version =max (set (versions ),key =versions .count )
        best_matches .sort (key =lambda x :x ['query_segment'])
        avg_sim =float (np .mean ([m ['similarity']for m in best_matches ]))
        max_sim =float (np .max ([m ['similarity']for m in best_matches ]))
        coverage =len (best_matches )/N_q 
        results .append ({
        'song_id':sid ,
        'version':version ,
        'matches':best_matches ,
        'match_count':len (best_matches ),
        'max_similarity':max_sim ,
        'avg_similarity':avg_sim ,
        'global_score':score ,
        'raw_global_score':score ,
        'coverage':coverage 
        })
    return results 
def search_per_segment (all_embs ,index ,top_k_per_seg =5 ):
    q_t =torch .tensor (all_embs ,dtype =torch .float32 ).to (DEVICE )
    q_t =F .normalize (q_t ,dim =-1 )
    idx_np =np .array (index ['embeddings'],dtype =np .float32 )
    norms =np .linalg .norm (idx_np ,axis =1 ,keepdims =True )
    norms [norms ==0 ]=1e-8 
    idx_np /=norms 
    idx_t =torch .from_numpy (idx_np ).to (DEVICE )
    meta =index ['metadata']
    N_q =all_embs .shape [0 ]
    with torch .no_grad ():
        sims =(q_t @idx_t .T ).cpu ().numpy ()
    results =[]
    for q_idx in range (N_q ):
        seg_sims =sims [q_idx ]
        top_indices =np .argsort (seg_sims )[::-1 ][:top_k_per_seg ]
        top_matches =[]
        for db_idx in top_indices :
            m =meta [db_idx ]
            top_matches .append ({
            'song_id':m ['song_id'],
            'version':m ['version'],
            'match_idx':m ['segment_idx'],
            'start_time':m ['start_time'],
            'end_time':m ['end_time'],
            'similarity':float (seg_sims [db_idx ]),
            'global_key':m .get ('global_key','Unknown'),
            })
        results .append ({'query_segment':q_idx ,'top_matches':top_matches })
    return results 
def rescale_similarity (raw_sim ):
    return max (0.0 ,min (1.0 ,(raw_sim -SIMILARITY_THRESHOLD )/(1.0 -SIMILARITY_THRESHOLD )))
def filter_consecutive_windows (results ,threshold =SIMILARITY_THRESHOLD ,
min_consecutive =MIN_CONSECUTIVE_MATCHES ,max_gap =2 ):
    if threshold <=0 or min_consecutive <=1 :
        return results 
    qualified =[]
    for result in results :
        matches =result .get ('matches',[])
        if not matches :
            continue 
        sorted_matches =sorted (matches ,key =lambda x :x ['query_segment'])
        above =[m for m in sorted_matches if m ['similarity']>=threshold ]
        if len (above )<min_consecutive :
            continue 
        best_chain =[]
        current_chain =[above [0 ]]
        for i in range (1 ,len (above )):
            prev_q =current_chain [-1 ]['query_segment']
            curr_q =above [i ]['query_segment']
            if curr_q -prev_q <=max_gap :
                current_chain .append (above [i ])
            else :
                if len (current_chain )>len (best_chain ):
                    best_chain =list (current_chain )
                current_chain =[above [i ]]
        if len (current_chain )>len (best_chain ):
            best_chain =list (current_chain )
        if len (best_chain )<min_consecutive :
            continue 
        chain_sims =[m ['similarity']for m in best_chain ]
        first_q =best_chain [0 ]['query_segment']
        last_q =best_chain [-1 ]['query_segment']
        window_dur =(last_q -first_q )*SEGMENT_HOP +SEGMENT_DURATION 
        entry =dict (result )
        entry ['matches']=best_chain 
        entry ['match_count']=len (best_chain )
        entry ['avg_similarity']=float (np .mean (chain_sims ))
        entry ['max_similarity']=float (np .max (chain_sims ))
        entry ['consecutive_count']=len (best_chain )
        entry ['window_duration']=window_dur 
        entry ['chain_start_time']=best_chain [0 ].get ('start_time',first_q *SEGMENT_HOP )
        entry ['chain_end_time']=best_chain [-1 ].get ('end_time',(last_q *SEGMENT_HOP +SEGMENT_DURATION ))
        qualified .append (entry )
    qualified .sort (key =lambda x :x ['avg_similarity'],reverse =True )
    return qualified 
def format_time(seconds):
    total_seconds = int(round(seconds))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

def format_song_summary(aggregated: list, query_segments: list, metadata: dict) -> str:
    n_total = len(query_segments)
    lines = []
    lines.append(f"\n{'═'*68}")
    lines.append(f"  📊 RINGKASAN DETEKSI PLAGIASI MELODI")
    lines.append(f"{'═'*68}")
    lines.append(f"  Lagu input  : {n_total} segmen (masing-masing ±{SEGMENT_DURATION:.0f}s, step {SEGMENT_HOP:.0f}s)")
    lines.append("")

    if not aggregated:
        lines.append("  ❌ Tidak ada lagu yang terdeteksi mirip di atas threshold.")
        lines.append(f"{'═'*68}")
        return "\n".join(lines)

    for i, match in enumerate(aggregated[:5]):
        s_id   = match['song_id']
        meta   = metadata.get(s_id, {'title': f'Song {s_id}', 'artist': 'Unknown',
                                      'artist_original': 'Unknown', 'artist_cover': 'Unknown'})
        title  = meta.get('title', f'Song {s_id}')
        artist = meta.get('artist_original', meta.get('artist', 'Unknown'))
        version = match.get('version', 'original')
        if version == 'cover' and meta.get('artist_cover', 'Unknown') != 'Unknown':
            artist = meta['artist_cover']

        matches       = match.get('matches', [])
        n_match       = len(matches)
        pct_match     = n_match / n_total * 100 if n_total > 0 else 0.0
        avg_sim       = match.get('avg_similarity', 0.0) * 100
        max_sim       = match.get('max_similarity', 0.0) * 100
        global_score  = match.get('melody_similarity', match.get('global_score', 0.0)) * 100

        positions = []
        for m in sorted(matches, key=lambda x: x.get('query_segment', 0)):
            q_idx  = m.get('query_segment', 0)
            q_st   = q_idx * SEGMENT_HOP
            q_en   = q_st + SEGMENT_DURATION
            m_st   = m.get('start_time', 0)
            m_en   = m.get('end_time', m_st + SEGMENT_DURATION)
            positions.append((q_st, q_en, m_st, m_en))

        merged = []
        for q_st, q_en, m_st, m_en in positions:
            if merged and (q_st - merged[-1][1]) <= SEGMENT_HOP * 1.5:
                merged[-1] = (merged[-1][0], q_en, merged[-1][2], m_en)
            else:
                merged.append((q_st, q_en, m_st, m_en))

        lines.append(f"  {'─'*64}")
        lines.append(f"  🎵 #{i+1}  {title}  —  {artist}")
        lines.append(f"       Segmen cocok : {n_match} dari {n_total} ({pct_match:.1f}%)")
        lines.append(f"       Skor akhir   : {global_score:.1f}%  "
                     f"(avg embed: {avg_sim:.1f}%, max embed: {max_sim:.1f}%)")
        lines.append("")

        if merged:
            lines.append(f"       Posisi matching di lagu input → lagu database:")
            for q_s, q_e, m_s, m_e in merged:
                lines.append(
                    f"         Input [{format_time(q_s)}–{format_time(q_e)}]"
                    f"  ↔  Database [{format_time(m_s)}–{format_time(m_e)}]"
                )
        lines.append("")

    lines.append(f"{'═'*68}")
    return "\n".join(lines)
def format_time_range (start ,end ):
    return f"{format_time(start)}-{format_time(end)}"
def get_similarity_tier (score ):
    if score >=0.70 :
        return "🔴 Very High"
    elif score >=0.45 :
        return "🟡 High"
    elif score >=0.20 :
        return "🟢 Moderate"
    else :
        return "⚪ Low"
def load_song_metadata (csv_path ="dataset.csv"):
    metadata ={}
    if not os .path .exists (csv_path ):
        return metadata 
    try :
        import csv 
        with open (csv_path ,'r',encoding ='utf-8')as f :
            reader =csv .reader (f )
            next (reader ,None )
            for parts in reader :
                if len (parts )<4 :
                    continue 
                s_id =parts [0 ].strip ()
                if s_id .isdigit ():
                    s_id =s_id .zfill (3 )
                if s_id not in metadata :
                    metadata [s_id ]={
                    'title':parts [2 ].strip (),
                    'artist':parts [3 ].strip (),
                    'artist_original':'Unknown',
                    'artist_cover':'Unknown'
                    }
                version =parts [4 ].strip ().lower ()if len (parts )>4 else 'original'
                if version =='original':
                    metadata [s_id ]['artist_original']=parts [3 ].strip ()
                    metadata [s_id ]['title']=parts [2 ].strip ()
                elif version =='cover':
                    metadata [s_id ]['artist_cover']=parts [3 ].strip ()
    except Exception :
        pass 
    return metadata 
def is_youtube_url (text ):
    youtube_patterns =[
    r'(https?://)?(www\.)?youtube\.com/watch\?v=',
    r'(https?://)?(www\.)?youtu\.be/',
    r'(https?://)?(www\.)?youtube\.com/shorts/',
    ]
    return any (re .search (p ,text )for p in youtube_patterns )
def download_youtube (url ,output_dir =None ):
    try :
        import yt_dlp 
    except ImportError :
        print ("❌ yt-dlp not installed. Run: pip install yt-dlp")
        return None 
    if output_dir is None :
        import shutil 
        output_dir =os .path .join (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),'downloads')
        if os .path .exists (output_dir ):
            shutil .rmtree (output_dir ,ignore_errors =True )
    os .makedirs (output_dir ,exist_ok =True )
    output_wav =os .path .join (output_dir ,'query_audio.wav')
    base_path =os .path .join (output_dir ,'query_audio')
    print (f"\n📥 Downloading from YouTube → {output_dir}...")
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
            print (f"✅ Downloaded: {output_wav}")
            return output_wav 
        for ext in ['wav','mp3','opus','m4a','webm']:
            alt =f"{base_path}.{ext}"
            if os .path .exists (alt ):
                print (f"✅ Downloaded: {alt}")
                return alt 
        print (f"❌ Audio file not found in {output_dir} after download")
        return None 
    except Exception as e :
        print (f"❌ Download error: {e}")
        return None 
def query (input_path ,model_path =None ,top_k =TOP_K_RESULTS ,threshold =SIMILARITY_THRESHOLD ,
override_key =None ):
    print ("="*70 )
    print ("🔍 MELODY SIMILARITY QUERY (Enhanced v3)")
    print ("="*70 )
    _original_path =input_path 
    _normalized_tmp =None 
    if is_youtube_url (input_path ):
        wav_path =download_youtube (input_path )
        if wav_path is None :
            return []
        input_path =wav_path 
    else :
        _normalized_tmp =normalize_query_audio (input_path )
        if _normalized_tmp !=input_path :
            input_path =_normalized_tmp 
    print ("\n🧠 Loading model...")
    model =load_model (model_path )
    print ("📂 Loading index...")
    index =load_index ()
    print (f"📊 Index contains {len(index['embeddings'])} segments")
    print ("\n🎵 Extracting features...")
    query_segments ,query_key =extract_query_features (input_path )
    key_confidence =query_segments [0 ].get ('key_confidence',0.0 )if query_segments else 0.0 
    if override_key :
        print (f"   ℹ️  Key overridden by user: {override_key} (was: {query_key})")
        query_key =override_key 
    if query_segments is None :
        print ("❌ Could not extract features from query file.")
        return []
    print (f"\n🔍 Searching {len(query_segments)} query segments (max-sim GPU)...")
    all_embs =extract_query_embeddings (model ,query_segments )
    aggregated =search_maxsim (all_embs ,index ,top_k =top_k )
    n_before =len (aggregated )
    aggregated =filter_consecutive_windows (aggregated ,threshold =threshold ,
    min_consecutive =MIN_CONSECUTIVE_MATCHES )
    print (f"   🔗 Consecutive filter: {n_before} candidates → {len(aggregated)} "
    f"(threshold: {threshold:.0%}, min consecutive: {MIN_CONSECUTIVE_MATCHES})")
    if aggregated :
        query_vocals =None 
        if _DEMUCS_AVAILABLE :
            try :
                from demucs_utils import separate_vocals 
                print ("\n🎹  Separating vocals (Demucs) for melody proof...")
                _raw_audio =load_audio (input_path )
                if _raw_audio is not None :
                    query_vocals =separate_vocals (_raw_audio ,SAMPLE_RATE )
                    if query_vocals is not None :
                        print ("   ✅ Vocals separated — melody proof uses pYIN (more accurate)")
                    else :
                        print ("   ⚠️  separate_vocals gagal — using CQT peak-picking")
            except Exception as _de :
                print (f"   ⚠️  Demucs vocals skipped ({_de}) — using CQT peak-picking")
        else :
            print ("   ⚠️  Demucs tidak tersedia — skipping vocal separation for Stage 2")
        top_candidates =aggregated [:20 ]
        if verify_candidates is not None :
            print ("\n🎼 Stage 2: Verifying melodic contour...")
            verify_candidates (
            query_path =input_path ,
            query_key =query_key ,
            candidates =top_candidates ,
            dataset_dir =DATASET_DIR ,
            query_segments =query_segments ,
            features_dir =FEATURES_DIR ,
            query_vocals =query_vocals ,
            )
        else :
            print ("\n⚠️  Stage 2 skipped — contour_similarity module not available.")
        for match in top_candidates :
            contour_details =[d for d in match .get ('contour_details',[])if d is not None ]
            scores_list =[d ['similarity']for d in contour_details ]
            if scores_list :
                max_contour =max (scores_list )
                n_strong =sum (1 for s in scores_list if s >=0.30 )
                bonus =min ((n_strong -1 )/3.0 ,1.0 )*0.25 if n_strong >1 else 0.0 
                melody_sim =min (max_contour *(1.0 +bonus ),1.0 )
                best_local_idx =int (np .argmax (scores_list ))
                match ['best_contour_detail']=contour_details [best_local_idx ]
                orig_matches =match .get ('matches',[])
                match ['best_contour_match']=orig_matches [best_local_idx ]if best_local_idx <len (orig_matches )else {}
                match ['max_contour_score']=max_contour 
                match ['n_strong_segments']=n_strong 
            else :
                melody_sim =match .get ('global_score',0.0 )*0.5 
                match ['max_contour_score']=0.0 
                match ['n_strong_segments']=0 
                match ['best_contour_detail']={}
                match ['best_contour_match']={}
            match ['melody_similarity']=melody_sim 
        top_candidates .sort (key =lambda x :x ['melody_similarity'],reverse =True )
        aggregated =top_candidates 
        print (f"   ✅ Contour verification complete. Found {len(aggregated)} candidates.")
    print ("\n"+"="*70 )
    print ("📊 QUERY RESULTS")
    print ("="*70 )
    print (f"\n🎵 Query: {os.path.basename(input_path)}")
    conf_str =f"{key_confidence:.2f}"
    warn_str ="  ⚠️ low confidence — might be relative major/minor"if key_confidence <0.30 else ""
    print (f"🔑 Detected Key: {query_key}  (confidence: {conf_str}{warn_str})")
    scale_notes =get_scale_notes (query_key )
    if scale_notes :
        print (f"🎹 Scale: {' '.join(scale_notes)}")
        print (f"   (verify: play these notes on keyboard while listening to the song)")
    if not aggregated :
        print ("\n❌ No similar melody found above threshold.")
    else :
        print (f"\n🎯 Found {len(aggregated)} potential matches:\n")
        metadata =load_song_metadata ()
        print(format_song_summary(aggregated, query_segments, metadata))
        for i ,match in enumerate (aggregated [:5 ]):

            s_id =match ['song_id']
            meta =metadata .get (s_id ,{
            'title':f'Song {s_id}','artist':'Unknown',
            'artist_cover':'Unknown','artist_original':'Unknown'
            })
            display_artist =meta ['artist']
            if match ['version']=='cover'and meta .get ('artist_cover')!='Unknown':
                display_artist =meta ['artist_cover']
            elif match ['version']=='original'and meta .get ('artist_original')!='Unknown':
                display_artist =meta ['artist_original']
            match_key =match ['matches'][0 ].get ('match_global_key','Unknown')if match .get ('matches')else 'Unknown'
            match_scale =get_scale_notes (match_key )
            melody_pct =match .get ('melody_similarity',0.0 )*100 
            max_c_pct =match .get ('max_contour_score',0.0 )*100 
            n_strong =match .get ('n_strong_segments',0 )
            tier =get_similarity_tier (match .get ('melody_similarity',0.0 ))
            avg_emb =match .get ('avg_similarity',0.0 )*100 
            max_emb =match .get ('max_similarity',0.0 )*100 
            consec =match .get ('consecutive_count',len (match .get ('matches',[])))
            win_dur =match .get ('window_duration',consec *SEGMENT_HOP +SEGMENT_DURATION -SEGMENT_HOP )
            chain_start =match .get ('chain_start_time',0.0 )
            chain_end =match .get ('chain_end_time',win_dur )
            print (f"{'─' * 64}")
            print (f"🎵 Rank #{i+1}: {meta['title']} ({display_artist})")
            print (f"   🧠 Embedding:          avg {avg_emb:.1f}%  max {max_emb:.1f}%")
            print (f"   🔗 Consecutive Match:  {consec} window(s) × {SEGMENT_DURATION:.0f}s  "
            f"[{format_time(chain_start)} → {format_time(chain_end)}]  ({win_dur:.0f}s total)")
            print (f"   🎼 Best Melody Window: {max_c_pct:.1f}%  {tier}")
            print (f"   🏅 Final Score:        {melody_pct:.1f}%  ({n_strong} window(s) ≥30% melody)")
            print (f"   🔑 Match Key: {match_key}",end ="")
            if match_scale :
                print (f" ({' '.join(match_scale)})")
            else :
                print ()
            best_detail =match .get ('best_contour_detail',{})
            best_m =match .get ('best_contour_match',{})
            if best_detail and isinstance (best_detail ,dict )and best_m :
                q_seg_idx =best_m .get ('query_segment',0 )
                q_st =q_seg_idx *SEGMENT_HOP 
                m_st =best_m .get ('start_time',0 )
                q_rng =format_time_range (q_st ,q_st +SEGMENT_DURATION )
                m_rng =format_time_range (m_st ,m_st +SEGMENT_DURATION )
                b_pct =best_detail .get ('similarity',0.0 )*100 
                q_notes =best_detail .get ('query_paired','(no melody)')[:60 ]
                m_notes =best_detail .get ('match_paired','(no melody)')[:60 ]
                q_sol =best_detail .get ('query_solfege','')[:45 ]
                src =best_detail .get ('source','cqt')
                print (f"\n   🏆 Best Melody Match ({b_pct:.1f}%)  [{src}]:")
                print (f"      Query [{q_rng}]  ↔  Match [{m_rng}]")
                print (f"      Query: {q_notes}")
                print (f"      Match: {m_notes}")
                if q_sol :
                    print (f"      Sol:   {q_sol}")
            _orig_matches =match .get ('matches',[])
            _contour_details =[d for d in match .get ('contour_details',[])if d is not None ]
            window_list =[]
            for k ,m in enumerate (_orig_matches ):
                cd =_contour_details [k ]if k <len (_contour_details )else None 
                mel =cd ['similarity']if cd else 0.0 
                window_list .append ((mel ,m ,cd ))
            window_list .sort (key =lambda x :x [0 ],reverse =True )
            if window_list :
                print (f"\n   📋 All windows (sorted by melody score):")
                for rank_w ,(mel_s ,m ,cd )in enumerate (window_list [:6 ]):
                    q_idx =m .get ('query_segment',0 )
                    q_st2 =q_idx *SEGMENT_HOP 
                    q_rng2 =format_time_range (q_st2 ,q_st2 +SEGMENT_DURATION )
                    m_rng2 =format_time_range (m .get ('start_time',0 ),m .get ('end_time',0 ))
                    emb_p =m .get ('similarity',0.0 )*100 
                    mel_p =mel_s *100 
                    mark ="  ★ best"if rank_w ==0 else ""
                    print (f"      [{q_rng2}] ↔ [{m_rng2}]  melody:{mel_p:4.0f}%  embed:{emb_p:4.0f}%{mark}")
            print ()
        print ("="*70 )
        ask_proof =input ("🎹 Generate melody proof files? (y/n) [n]: ").strip ().lower ()
        if ask_proof =='y':
            query_name =os .path .splitext (os .path .basename (input_path ))[0 ]
            _root_dir =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
            proof_base =os .path .join (_root_dir ,'proof',query_name )
            print (f"\n📂 Generating proof in: {proof_base}")
            for i ,match in enumerate (aggregated [:5 ]):
                s_id =match ['song_id']
                meta =metadata .get (s_id ,{'title':f'Song {s_id}'})
                safe_title ="".join (c if c .isalnum ()or c in ' _-'else '_'for c in meta ['title'])
                rank_folder =os .path .join (proof_base ,f"rank_{i+1}_{safe_title}")
                os .makedirs (rank_folder ,exist_ok =True )
                sorted_matches =sorted (match ['matches'],key =lambda x :x ['query_segment'])
                _orig_matches =match ['matches']
                _contour_details =[d for d in match .get ('contour_details',[])if d is not None ]
                _contour_by_qs ={
                m ['query_segment']:_contour_details [k ]
                for k ,m in enumerate (_orig_matches )
                if k <len (_contour_details )
                }
                for j ,m in enumerate (sorted_matches ):
                    q_seg_idx =m ['query_segment']
                    q_start =q_seg_idx *SEGMENT_HOP 
                    q_end =q_start +SEGMENT_DURATION 
                    m_start =m ['start_time']
                    m_end =m ['end_time']
                    q_paired ="(no melody detected)"
                    m_paired ="(no melody detected)"
                    q_solfege =""
                    m_solfege =""
                    contour_sim =0.0 
                    cd =_contour_by_qs .get (m ['query_segment'])
                    if cd :
                        contour_sim =cd .get ('similarity',0.0 )*100 
                        if cd .get ('query_paired'):
                            q_paired =cd ['query_paired']
                        if cd .get ('match_paired'):
                            m_paired =cd ['match_paired']
                        if cd .get ('query_solfege'):
                            q_solfege =cd ['query_solfege']
                        if cd .get ('match_solfege'):
                            m_solfege =cd ['match_solfege']
                    txt_path =os .path .join (rank_folder ,f"segment_{j+1}_melody.txt")
                    with open (txt_path ,'w',encoding ='utf-8')as f :
                        f .write (f"Melody Similarity Proof\n")
                        f .write (f"{'=' * 50}\n\n")
                        f .write (f"Melody Similarity: {contour_sim:.1f}%\n\n")
                        f .write (f"QUERY: {os.path.basename(input_path)}\n")
                        f .write (f"  Timestamp: {format_time(q_start)} - {format_time(q_end)}\n")
                        f .write (f"  Key: {query_key}\n")
                        f .write (f"  Melody (Notes):   {q_paired}\n")
                        f .write (f"  Melody (Solfege): {q_solfege}\n\n")
                        f .write (f"MATCH: {meta['title']}\n")
                        f .write (f"  Timestamp: {format_time(m_start)} - {format_time(m_end)}\n")
                        f .write (f"  Key: {match['matches'][0].get('match_global_key', 'Unknown')}\n")
                        f .write (f"  Melody (Notes):   {m_paired}\n")
                        f .write (f"  Melody (Solfege): {m_solfege}\n")
                print (f"   ✅ Rank {i+1}: {meta['title']} — {len(sorted_matches)} segments saved")
            print (f"\n📂 Proof files saved to: {os.path.abspath(proof_base)}")
    cleanup_normalized_audio (_normalized_tmp ,_original_path )if _normalized_tmp else None 
    return aggregated 
def query_segment_mode (input_path ,model_path =None ,top_k_per_seg =5 ,
verify_melody =False ):
    print ("="*70 )
    print ("\U0001f50d MELODY SIMILARITY \u2014 SEGMENT MODE")
    print ("="*70 )
    _original_path_seg =input_path 
    _normalized_tmp_seg =None 
    if is_youtube_url (input_path ):
        wav_path =download_youtube (input_path )
        if wav_path is None :
            return 
        input_path =wav_path 
    else :
        _normalized_tmp_seg =normalize_query_audio (input_path )
        if _normalized_tmp_seg !=input_path :
            input_path =_normalized_tmp_seg 
    print ("\n\U0001f9e0 Loading model...")
    model =load_model (model_path )
    print ("\U0001f4c2 Loading index...")
    index =load_index ()
    print (f"\U0001f4ca Index contains {len(index['embeddings'])} segments")
    print ("\n\U0001f3b5 Extracting features...")
    query_segments ,query_key =extract_query_features (input_path )
    if query_segments is None :
        print ("\u274c Could not extract features from query file.")
        return 
    key_confidence =query_segments [0 ].get ('key_confidence',0.0 )if query_segments else 0.0 
    conf_str =f"{key_confidence:.2f}"
    warn_str ="  \u26a0\ufe0f low confidence"if key_confidence <0.30 else ""
    print (f"\n\U0001f3b5 Query: {os.path.basename(input_path)}")
    print (f"\U0001f511 Key: {query_key}  (confidence: {conf_str}{warn_str})")
    print (f"\U0001f4e6 {len(query_segments)} segments \u00d7 {SEGMENT_DURATION:.0f}s each  "
    f"(hop {SEGMENT_HOP:.0f}s)")
    all_embs =extract_query_embeddings (model ,query_segments )
    query_vocals =None 
    if verify_melody :
        if _DEMUCS_AVAILABLE :
            try :
                from demucs_utils import separate_vocals 
                print ("\n🏙️  Separating vocals+melody (Demucs) for verification...")
                _raw =load_audio (input_path )
                if _raw is not None :
                    query_vocals =separate_vocals (_raw ,SAMPLE_RATE )
                    if query_vocals is not None :
                        print ("   ✅ Done — melody verification uses pYIN")
                    else :
                        print ("   ⚠️  separate_vocals gagal — skipping melody verify")
            except Exception as _de :
                print (f"   ⚠️  Demucs skipped ({_de})")
        else :
            print ("   ⚠️  Demucs tidak tersedia — skipping melody verify")
    print (f"\n\U0001f50d Searching {len(query_segments)} segments independently "
    f"(top-{top_k_per_seg} each, no thresholds)...")
    seg_results =search_per_segment (all_embs ,index ,
    top_k_per_seg =top_k_per_seg )
    meta_db =load_song_metadata ()
    print ("\n"+"="*70 )
    print ("\U0001f4ca PER-SEGMENT RESULTS")
    print ("="*70 )
    print ("   Each row = one 15s query window \u2192 its best DB match")
    print ("   (no thresholds applied \u2014 every segment shown)\n")
    for sr_item in seg_results :
        q_idx =sr_item ['query_segment']
        q_start =q_idx *SEGMENT_HOP 
        q_rng =format_time_range (q_start ,q_start +SEGMENT_DURATION )
        top =sr_item ['top_matches']
        if not top :
            print (f"  Query [{q_rng}]  \u2192  (no matches in index)")
            continue 
        best =top [0 ]
        s_id =best ['song_id']
        meta =meta_db .get (s_id ,{'title':f'Song {s_id}','artist':'Unknown',
        'artist_original':'Unknown',
        'artist_cover':'Unknown'})
        da =(meta .get ('artist_original')or meta ['artist'])if best ['version']=='original'else (meta .get ('artist_cover')or meta ['artist'])
        m_rng =format_time_range (best ['start_time'],best ['end_time'])
        emb_pct =best ['similarity']*100 
        melody_str =""
        if verify_melody and query_vocals is not None :
            try :
                from contour_similarity import compare_audio_cqt_segments ,_load_match_cqt 
                shift_to_c =query_segments [q_idx ].get ('shift_to_c',0 )
                m_cqt =_load_match_cqt (best ['song_id'],best ['version'],
                best ['match_idx'])
                if m_cqt is not None :
                    q_s =int (q_idx *SEGMENT_HOP *SAMPLE_RATE )
                    q_e =int (q_s +SEGMENT_DURATION *SAMPLE_RATE )
                    q_v =query_vocals [q_s :min (q_e ,len (query_vocals ))]
                    if len (q_v )>=int (SAMPLE_RATE *0.3 ):
                        cd =compare_audio_cqt_segments (
                        q_v ,m_cqt ,shift_back_semitones =-shift_to_c )
                        melody_str =f"  melody: {cd['similarity']*100:.0f}%"
            except Exception :
                pass 
        print (f"  Query [{q_rng}]  \u2192  {meta['title']} ({da})"
        f"  [{m_rng}]  embed: {emb_pct:.0f}%{melody_str}")
        for alt in top [1 :3 ]:
            s2 =alt ['song_id']
            m2 =meta_db .get (s2 ,{'title':f'Song {s2}','artist':'Unknown',
            'artist_original':'Unknown',
            'artist_cover':'Unknown'})
            da2 =(m2 .get ('artist_original')or m2 ['artist'])if alt ['version']=='original'else (m2 .get ('artist_cover')or m2 ['artist'])
            mr2 =format_time_range (alt ['start_time'],alt ['end_time'])
            print (f"       alt \u2192 {m2['title']} ({da2})  [{mr2}]"
            f"  embed: {alt['similarity']*100:.0f}%")
    print ("\n"+"="*70 )
    if _normalized_tmp_seg is not None :
        cleanup_normalized_audio (_normalized_tmp_seg ,_original_path_seg )
def interactive_mode ():
    print ("\n"+"="*70 )
    print ("\U0001f3b5 MELODY SIMILARITY - INTERACTIVE MODE")
    print ("="*70 )
    print ("   Commands:  <file/URL>   query a song")
    print ("              mode         toggle search mode (standard \u2194 segment)")
    print ("              q            quit")
    use_segment =False 
    while True :
        mode_tag ="segment"if use_segment else "standard"
        choice =input (f"\n\U0001f522 [{mode_tag}] Enter path/URL (or 'q'/'mode'): ").strip ()
        if choice .lower ()in ['q','quit']:
            break 
        if choice .lower ()=='mode':
            use_segment =not use_segment 
            label ="segment"if use_segment else "standard"
            print (f"   \u21a9  Mode \u2192 {label}")
            continue 
        path =choice .strip ('"')
        if is_youtube_url (path ):
            if use_segment :
                query_segment_mode (path )
            else :
                query (path )
            continue 
        if not os .path .exists (path ):
            print ("\u274c File not found")
            continue 
        if use_segment :
            query_segment_mode (path )
        else :
            query (path )
if __name__ =="__main__":
    parser =argparse .ArgumentParser ()
    parser .add_argument ("input",nargs ="?")
    parser .add_argument ("--top-k",type =int ,default =TOP_K_RESULTS )
    parser .add_argument ("--threshold",type =float ,default =SIMILARITY_THRESHOLD )
    parser .add_argument ("--key",type =str ,default =None ,
    help ="Override auto-detected key, e.g. 'E Minor' or 'G Major'. "
    "Use when auto-detection seems wrong.")
    parser .add_argument ("--interactive","-i",action ="store_true")
    parser .add_argument ("--segment-mode","-s",action ="store_true",
    help ="Per-segment mode: each 15s window is searched independently. "
    "No thresholds, no aggregation. Best for partial-match detection.")
    parser .add_argument ("--verify",action ="store_true",
    help ="In --segment-mode: also run pYIN melody verification "
    "on the top match of each segment (slower).")
    args =parser .parse_args ()
    if args .interactive or not args .input :
        interactive_mode ()
    elif args .input :
        if args .segment_mode :
            query_segment_mode (args .input ,
            top_k_per_seg =min (args .top_k ,5 ),
            verify_melody =args .verify )
        else :
            query (args .input ,top_k =args .top_k ,threshold =args .threshold ,
            override_key =args .key )
    else :
        interactive_mode ()

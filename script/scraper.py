import os 
import csv 
import re 
import time 
import urllib .parse 
import pandas as pd 
from yt_dlp import YoutubeDL 
import sys 
sys .stdout .reconfigure (encoding ='utf-8')
import shutil 
import numpy as np 
import librosa 
import threading 
import gc 
import signal 
from concurrent .futures import ThreadPoolExecutor 
PRINT_LOCK =threading .Lock ()
CSV_LOCK =threading .Lock ()
DICT_LOCK =threading .Lock ()
REALTIME_ID_LOCK =threading .Lock ()
API_KEY_LOCK =threading .Lock ()
SHUTDOWN_EVENT =threading .Event ()
def handle_interrupt (*args ):
    print ("\n   🛑 Shutdown signal received! Stopping all tasks and cleaning up...",flush =True )
    SHUTDOWN_EVENT .set ()
    try :
        cleanup_temp_folders ()
    except Exception :
        pass 
    os ._exit (1 )
signal .signal (signal .SIGINT ,handle_interrupt )
signal .signal (signal .SIGTERM ,handle_interrupt )
import uuid 
realtime_gap_queue =[]
realtime_next_new_id =1 
def init_realtime_id_pool (gap_ids ,next_new_id ):
    global realtime_gap_queue ,realtime_next_new_id 
    realtime_gap_queue =list (gap_ids )
    realtime_next_new_id =int (next_new_id )
def get_realtime_folder_id ():
    global realtime_gap_queue ,realtime_next_new_id 
    with REALTIME_ID_LOCK :
        if realtime_gap_queue :
            folder_id =realtime_gap_queue .pop (0 )
        else :
            folder_id =realtime_next_new_id 
            realtime_next_new_id +=1 
        return folder_id 
import json 
import urllib .request 
import urllib .error 
API_KEYS =[
"AIzaSyABN0_082VVIN5a7ujMLy6-ZcJ6P5ZymVQ",
"AIzaSyAR0SGCOgySnPZld_tYTBEM_ZQct6VqOZo",
"AIzaSyAJ_uAOeIRcUc-hgm3_wJvVyaLtGj3qris",
"AIzaSyCevSEu2rTb-zhw7av6Fe5pGeNhaDNzMoU",
"AIzaSyBGVRwPdrkxTde9FZPLWMWvHpCSCM22SOA",
"AIzaSyBvvOjxDU3bBvh60VoVLQd07C7hYAgo2pk",
"AIzaSyCAJvKM9Cbmc0_bDWQF4PpWrS-t9KKsgw0",
"AIzaSyBRuicuzdqsAMrUqHfRSHZ-tx3GIVsyT7c",
"AIzaSyBBga9Vu-Rz1JUNTWscbHR8A5bUmUQmm3o",
"AIzaSyAmmUtExHy4TfXCKzOSagX3wEgKrIiV1c0",
"AIzaSyAvYy1A4_zx3P0kFnb-P7IpYellkpMtJfQ",
"AIzaSyCdwtqOUN5F4URsmdZtYhZmHCNGdjKIWco",
"AIzaSyCQiXVLXj47ax7uaeXAwDpWplmHgFsi8RY",
"AIzaSyAmqbq6exqHoTWYySZbItuXk5ofPGBan9M",
"AIzaSyB2CRPTfZT4Be28DFHoJipA4FYE8M_pGYs",
"AIzaSyDx60Nrnmk1y_7HtPrYMARpGgvStqPPD9g",
"AIzaSyC-08zbpOUl8TTdBvHDeSABaJUVqUnbAvY",
"AIzaSyB_v2aa7OW29ovXjbu4GB6c9ktcrYm8Epw",
"AIzaSyBS8ZaR1dCQKe2zLGA2HTl3ooBTovb3j_g",
"AIzaSyCXxTbuMK-Y-VxkFmmxK6TlVVwqQ-HJ24E",
"AIzaSyApepxSByHkKGREpUc0M_EUxaUlbe7QSA4",
"AIzaSyCDCkuYq-yf3eaWHbLHFn3s8OeHW0793xU",
"AIzaSyC3K5HuzhG-JwUEIcZFW2ICO25A4VOFyo4",
"AIzaSyAr7DyIr2QGZhPZlouy29B0sFtgTGcyc8k",
"AIzaSyBMCrIMIW_CKFHNegiEAfaVeRRws8xfg10",
"AIzaSyB5sAPbAhNrfPHMfsSLesaWrxLHPcGuFQQ",
"AIzaSyDv7mEkTed2LAekI3dZZ43vES7JNbSH3CU",
"AIzaSyAv51uvsg2naNukrW269uvxe0f_zfxQIHE",
"AIzaSyDgh_AOn1UqfzEwbg5S4jTWnSBgIBxhhx0",
"AIzaSyAw5LKtD8zRX81zjzqujepjlvqMziebhqo",
"AIzaSyAyUzBjrkt4KMEAl9ahqimToaRm1T421Mo",
"AIzaSyB6tVOrSecEDMqv_j805v9pTHhUfJ5NoN0",
"AIzaSyCTHhwvvS50WDfGHy4fOqWrKcWgbM-7WIs",
"AIzaSyDQWefzdJe3LJsZ3iajh14-ojHdjTaWU20",
"AIzaSyB4L1hpw3il4j072CgSvpcLq1jnlVt4g6A",
"AIzaSyChcnG1iS1ZCVbOtlCuqo2kVW3Xjy81qog",
"AIzaSyDCNhqkbn8nabiqwx2mVAOz-c3-jbDxUmI",
"AIzaSyDG91BytrzczTNJoyCoFCGm55XzmKTXl48"
]
class QuotaExceededError (Exception ):
    pass 
_orig_print =print 
def safe_print (*args ,**kwargs ):
    with PRINT_LOCK :
        _orig_print (*args ,**kwargs )
print =safe_print 
API_STATE ={"index":0 }
channel_counts :dict ={}
def sanitize_spotify_title (title :str )->str :
    clean =re .sub (r'\(.*?\)','',title )
    clean =re .sub (r'\[.*?\]','',clean )
    clean =re .split (r' - |- |: ',clean )[0 ]
    return clean .strip ()
def parse_yt_duration (duration_str ):
    if not duration_str :
        return 0 
    match =re .match (r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',duration_str )
    if not match :
        return 0 
    h ,m ,s =match .groups ()
    return int (h or 0 )*3600 +int (m or 0 )*60 +int (s or 0 )
def search_youtube_api (query ,max_results =30 ,category_id =10 ):
    while True :
        with API_KEY_LOCK :
            idx =API_STATE ["index"]
            if idx >=len (API_KEYS ):
                break 
            key =API_KEYS [idx ]
        url =f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults={max_results}&q={urllib.parse.quote(query)}&type=video&videoCategoryId={category_id}&key={key}"
        try :
            req =urllib .request .Request (url )
            with urllib .request .urlopen (req ,timeout =10 )as response :
                data =json .loads (response .read ().decode ())
                return [f"https://www.youtube.com/watch?v={item['id']['videoId']}"for item in data .get ('items',[])]
        except urllib .error .HTTPError as e :
            if e .code ==403 :
                with API_KEY_LOCK :
                    if API_STATE ["index"]==idx :
                        print (f"\n      🚨 YouTube API Key #{idx + 1} QUOTA EXCEEDED! Switching to the next key...")
                        API_STATE ["index"]+=1 
                continue 
            else :
                print (f"      ⚠️ YouTube API HTTP Error: {e.code} - {e.reason}")
                return []
        except Exception as e :
            print (f"      ⚠️ YouTube API Exception: {e}")
            return []
    print (f"\n      🚨 ALL {len(API_KEYS)} API KEYS DEPLETED.")
    print (f"      You have maxed out your daily quota for all keys. Halting the script.")
    raise QuotaExceededError ("All YouTube API keys exhausted.")
def search_youtube_api_with_metadata (query ,max_results =30 ,category_id =10 ):
    while True :
        with API_KEY_LOCK :
            idx =API_STATE ["index"]
            if idx >=len (API_KEYS ):
                break 
            key =API_KEYS [idx ]
        search_url =f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults={max_results}&q={urllib.parse.quote(query)}&type=video&videoCategoryId={category_id}&key={key}"
        try :
            req =urllib .request .Request (search_url )
            with urllib .request .urlopen (req ,timeout =10 )as response :
                search_data =json .loads (response .read ().decode ())
                video_ids =[item ['id']['videoId']for item in search_data .get ('items',[])if item .get ('id',{}).get ('videoId')]
            if not video_ids :
                return []
            ids_str =",".join (video_ids )
            videos_url =f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={ids_str}&key={key}"
            req_vid =urllib .request .Request (videos_url )
            with urllib .request .urlopen (req_vid ,timeout =10 )as resp_vid :
                vid_data =json .loads (resp_vid .read ().decode ())
            results =[]
            for item in vid_data .get ('items',[]):
                vid_id =item ['id']
                snippet =item .get ('snippet',{})
                content =item .get ('contentDetails',{})
                stats =item .get ('statistics',{})
                duration_str =content .get ('duration','')
                view_count =int (stats .get ('viewCount',0 ))if stats .get ('viewCount')else 0 
                results .append ({
                "id":vid_id ,
                "webpage_url":f"https://www.youtube.com/watch?v={vid_id}",
                "title":snippet .get ('title',''),
                "uploader":snippet .get ('channelTitle',''),
                "channel":snippet .get ('channelTitle',''),
                "channel_id":snippet .get ('channelId',''),
                "duration":parse_yt_duration (duration_str ),
                "view_count":view_count ,
                "channel_is_verified":False 
                })
            return results 
        except urllib .error .HTTPError as e :
            if e .code in (403 ,429 ):
                with API_KEY_LOCK :
                    if API_STATE ["index"]==idx :
                        print (f"\n      🚨 YouTube API Key #{idx + 1} QUOTA EXCEEDED! Switching to the next key...")
                        API_STATE ["index"]+=1 
                continue 
            else :
                print (f"      ⚠️ YouTube API HTTP Error: {e.code} - {e.reason}")
                return []
        except Exception as e :
            print (f"      ⚠️ YouTube API Exception: {e}")
            return []
    print (f"\n      🚨 ALL {len(API_KEYS)} API KEYS DEPLETED.")
    raise QuotaExceededError ("All YouTube API keys exhausted.")
def get_channel_ids_for_urls (urls ):
    video_ids =[]
    for url in urls :
        vid =_youtube_video_id (url )
        if vid and vid !=url :
            video_ids .append (vid )
    if not video_ids :
        return []
    while True :
        with API_KEY_LOCK :
            idx =API_STATE ["index"]
            if idx >=len (API_KEYS ):
                break 
            key =API_KEYS [idx ]
        try :
            ids_str =",".join (video_ids )
            videos_url =f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={ids_str}&key={key}"
            req_vid =urllib .request .Request (videos_url )
            with urllib .request .urlopen (req_vid ,timeout =10 )as resp_vid :
                vid_data =json .loads (resp_vid .read ().decode ())
            channel_ids =[item .get ('snippet',{}).get ('channelId')for item in vid_data .get ('items',[])]
            return [cid for cid in channel_ids if cid ]
        except urllib .error .HTTPError as e :
            if e .code in (403 ,429 ):
                with API_KEY_LOCK :
                    if API_STATE ["index"]==idx :
                        print (f"\n      🚨 YouTube API Key #{idx + 1} QUOTA EXCEEDED! Switching to the next key...")
                        API_STATE ["index"]+=1 
                continue 
            else :
                return []
        except Exception :
            return []
    return []

SCRIPT_DIR =os .path .dirname (os .path .abspath (__file__ ))
PROJECT_ROOT =os .path .dirname (SCRIPT_DIR )
BASE_DIR =PROJECT_ROOT 
DATASET_CSV_PATH =os .path .join (PROJECT_ROOT ,"dataset.csv")
SPOTIFY_CSV_PATH =os .path .join (BASE_DIR ,"spotify_songs.csv")
EXHAUSTED_CSV_PATH =os .path .join (PROJECT_ROOT ,"exhausted_songs.csv")
DATASET_DIR =os .path .join (PROJECT_ROOT ,"dataset")
os .makedirs (DATASET_DIR ,exist_ok =True )
def _sanitize_query (s ):
    s =re .sub (r'["\+\&\|\(\)\[\]_]'," ",s )
    return re .sub (r"\s{2,}"," ",s ).strip ()
def _youtube_video_id (url :str )->str :
    if not url :
        return url 
    if "youtu.be/"in url :
        return url .split ("youtu.be/")[-1 ].split ("?")[0 ].split ("&")[0 ]
    parsed =urllib .parse .urlparse (url )
    vid =urllib .parse .parse_qs (parsed .query ).get ("v",[""])[0 ]
    if not vid and "/shorts/"in parsed .path :
        vid =parsed .path .split ("/shorts/")[-1 ].split ("/")[0 ]
    return vid or url 
def safe_delete_folder (folder_path ,retries =3 ,delay =2 ):
    if not os .path .exists (folder_path ):
        return 
    gc .collect ()
    for attempt in range (retries ):
        try :
            shutil .rmtree (folder_path ,ignore_errors =False )
            return 
        except Exception as e :
            sleep_time =delay if attempt <retries -1 else max (delay ,5 )
            time .sleep (sleep_time )
    try :
        shutil .rmtree (folder_path ,ignore_errors =True )
        if os .path .exists (folder_path ):
            print (
            f"      ⚠️ Warning: Could not fully delete {folder_path} after {retries} retries — folder may be locked by Windows."
            )
    except Exception :
        pass 
def cleanup_temp_folders ():
    if not os .path .exists (DATASET_DIR ):
        return 
    deleted_count =0 
    for dirname in os .listdir (DATASET_DIR ):
        if dirname .startswith ("temp_"):
            full_path =os .path .join (DATASET_DIR ,dirname )
            safe_delete_folder (full_path ,retries =1 ,delay =0.5 )
            deleted_count +=1 
    if deleted_count >0 :
        print (f"   🧹 Cleaned up {deleted_count} orphaned temporary folder(s).")
def safe_remove_file (file_path ,retries =3 ,delay =1.5 ):
    if not os .path .exists (file_path ):
        return 
    for attempt in range (retries ):
        try :
            os .remove (file_path )
            return 
        except Exception as e :
            if attempt <retries -1 :
                time .sleep (delay )
            else :
                print (f"      ⚠️ Warning: Could not remove file {file_path}: {e}")
def get_existing_songs ():
    existing =set ()
    highest_id =0 
    csv_ids :set =set ()
    if os .path .exists (DATASET_CSV_PATH ):
        try :
            with open (DATASET_CSV_PATH ,"r",encoding ="utf-8")as f :
                reader =csv .reader (f )
                next (reader ,None )
                for row in reader :
                    if len (row )>=5 :
                        s_id ,_ ,title ,artist ,version =tuple (row [:5 ])
                        if version .lower ()=="original":
                            _t =re .sub (r"[^a-z0-9 ]","",title .lower ().strip ())
                            _a =re .sub (r"[^a-z0-9 ]","",artist .lower ().strip ())
                            existing .add ((_a ,_t ))
                        try :
                            int_id =int (s_id )
                            csv_ids .add (int_id )
                            if int_id >highest_id :
                                highest_id =int_id 
                        except ValueError :
                            pass 
        except Exception as e :
            print (f"⚠️ Warning reading dataset.csv: {e}")
    if os .path .exists (EXHAUSTED_CSV_PATH ):
        try :
            with open (EXHAUSTED_CSV_PATH ,"r",encoding ="utf-8")as f :
                reader =csv .reader (f )
                next (reader ,None )
                for row in reader :
                    if len (row )>=2 :
                        artist ,title =tuple (row [:2 ])
                        _a =re .sub (r"[^a-z0-9 ]","",artist .lower ().strip ())
                        _t =re .sub (r"[^a-z0-9 ]","",title .lower ().strip ())
                        existing .add ((_a ,_t ))
        except Exception as e :
            print (f"⚠️ Warning reading exhausted_songs.csv: {e}")
    all_known =csv_ids 
    if all_known :
        highest_id =max (highest_id ,max (all_known ))
    gap_ids =[i for i in range (1 ,highest_id +1 )if i not in all_known ]
    orphan_ids =[]
    with CSV_LOCK :
        exhausted_count =0 
        if os .path .exists (EXHAUSTED_CSV_PATH ):
            with open (EXHAUSTED_CSV_PATH ,"r",encoding ="utf-8")as f :
                exhausted_count =max (0 ,sum (1 for line in f )-1 )
        dataset_count =len (existing )-exhausted_count 
        print (f"📊 Tracking {len(existing)} unique songs ({dataset_count} successfully downloaded, {exhausted_count} blacklisted/exhausted).")
        if gap_ids :
            print (f"♻️  Found {len(gap_ids)} empty ID slots from deleted/duplicate data that will be reused.")
    return existing ,highest_id ,gap_ids ,orphan_ids 
def get_candidate_songs (existing_songs ):
    if not os .path .exists (SPOTIFY_CSV_PATH ):
        print (f"❌ Error: Could not find {SPOTIFY_CSV_PATH}")
        sys .exit (1 )
    print ("📚 Loading Spotify song database...")
    df =pd .read_csv (SPOTIFY_CSV_PATH ,encoding ='utf-8')
    if 'track_name'in df .columns and 'track_artist'in df .columns :
        df =df .rename (columns ={'track_name':'Track','track_artist':'Artist'})
    elif 'Artist'not in df .columns or 'Track'not in df .columns :
        print ("❌ Error: Spotify dataset missing required columns!")
        print (f"   Available columns: {list(df.columns)}")
        sys .exit (1 )
    df =df .dropna (subset =['Artist','Track']).drop_duplicates (subset =['Artist','Track'])
    df ["Track"]=df ["Track"].astype (str ).str .strip ().str .replace (r"\s+"," ",regex =True )
    df ["Artist"]=df ["Artist"].astype (str )
    df ["_key"]=list (
    zip (
    df ["Artist"].str .lower ().str .strip ().str .replace (r"[^a-z0-9 ]","",regex =True ),
    df ["Track"].str .lower ().str .strip ().str .replace (r"[^a-z0-9 ]","",regex =True )
    )
    )
    mask =~df ["_key"].isin (existing_songs )
    filtered =df [mask ].drop (columns =["_key"])
    sampled =filtered .sample (frac =1 ).reset_index (drop =True )
    candidates =list (zip (sampled ["Artist"].str .strip (),sampled ["Track"].str .strip ()))
    return candidates 
def download_audio (query ,output_path ,is_original =True ,artist_name ="",song_title =""):
    ydl_opts ={
    "format":"bestaudio/best",
    "postprocessors":[
    {
    "key":"FFmpegExtractAudio",
    "preferredcodec":"wav",
    }
    ],
    "outtmpl":output_path .replace (".wav",""),
    "quiet":True ,
    "no_warnings":True ,
    "default_search":"ytsearch5",
    "socket_timeout":15 ,
    "retries":3 ,
    "ignoreerrors":True ,
    "nopart":True ,
    }
    _filter_fn =get_yt_dlp_filter (is_original ,artist_name ,song_title )
    try :
        with YoutubeDL (ydl_opts )as ydl :
            print (f"   🔍 Searching YouTube API: '{query}'")
            api_results =search_youtube_api_with_metadata (query ,max_results =5 ,category_id =10 )
            valid_entry =None 
            rejection_log =[]
            if api_results :
                for entry in api_results :
                    if not entry :
                        continue 
                    try :
                        reason =_filter_fn (entry )
                        if reason is None :
                            valid_entry =entry 
                            break 
                        else :
                            rejection_log .append (
                            (
                            entry .get ("title","?"),
                            entry .get ("view_count","?"),
                            entry .get ("duration","?"),
                            reason ,
                            )
                            )
                    except Exception :
                        pass 
            if not valid_entry :
                if rejection_log :
                    print (
                    f"   ⚠️ All {len(rejection_log)} candidates rejected. Reasons:"
                    )
                    for i ,(vtitle ,views ,dur ,reason )in enumerate (rejection_log ):
                        print (
                        f"      [{i+1}] '{vtitle}' | views={views} | {dur}s → {reason}"
                        )
                else :
                    print ("   ⚠️ YouTube returned 0 results for this query.")
                raise Exception ("No videos passed the quality filter.")
            video_url =valid_entry .get ("webpage_url","")
            uploader_name =valid_entry .get ("uploader","Unknown Channel")
            video_title =valid_entry .get ("title","")
            try :
                ydl .download ([video_url ])
            except Exception as dl_err :
                err_str =str (dl_err )
                output_path_str_check =str (output_path )
                expected_check =output_path_str_check .replace (".wav","")+".wav"
                file_ok =os .path .exists (output_path_str_check )or os .path .exists (
                expected_check 
                )
                if "WinError 32"in err_str and file_ok :
                    print (
                    f"      ⚠️ Non-fatal WinError 32 during FFmpeg cleanup (file exists, continuing)."
                    )
                else :
                    raise 
            output_path_str =str (output_path )
            expected_wav =output_path_str .replace (".wav","")+".wav"
            if os .path .exists (expected_wav )and os .path .abspath (
            expected_wav 
            )!=os .path .abspath (output_path_str ):
                try :
                    os .rename (expected_wav ,output_path_str )
                except Exception :
                    time .sleep (1.5 )
                    try :
                        os .rename (expected_wav ,output_path_str )
                    except Exception as try2_err :
                        print (
                        f"      ⚠️ Warning: Could not rename {expected_wav}: {try2_err}"
                        )
            return (
            os .path .exists (output_path_str ),
            video_url ,
            uploader_name ,
            video_title ,
            )
    except Exception as e :
        if isinstance (e ,QuotaExceededError ):
            raise 
        print (f"   ❌ Failed to download '{query}': {e}")
        return False ,"","",""
def _download_url (url :str ,output_path :str )->bool :
    ydl_opts ={
    "format":"bestaudio/best",
    "postprocessors":[{"key":"FFmpegExtractAudio","preferredcodec":"wav"}],
    "outtmpl":str (output_path ).replace (".wav",""),
    "quiet":True ,
    "no_warnings":True ,
    "socket_timeout":15 ,
    "retries":3 ,
    "ignoreerrors":True ,
    "nopart":True ,
    }
    try :
        with YoutubeDL (ydl_opts )as ydl :
            try :
                ydl .download ([url ])
            except Exception as dl_err :
                err_str =str (dl_err ).lower ()
                if any (kw in err_str for kw in ["sign in","bot","429","too many","rate limit","captcha"]):
                    raise Exception (f"BOT_DETECTED: {dl_err}")
                if "WinError 32"not in err_str :
                    raise 
            ps =str (output_path )
            expected =ps .replace (".wav","")+".wav"
            if os .path .exists (expected )and os .path .abspath (expected )!=os .path .abspath (ps ):
                try :
                    os .rename (expected ,ps )
                except Exception :
                    time .sleep (1.5 )
                    try :
                        os .rename (expected ,ps )
                    except Exception :
                        pass 
            return os .path .exists (ps )
    except Exception as e :
        if "BOT_DETECTED"in str (e ):
            raise 
        print (f"      ❌ Direct download failed: {e}")
        return False 
def get_yt_dlp_filter (is_original ,artist_name ="",song_title =""):
    song_title_lower =song_title .lower ()
    artist_name_lower =artist_name .lower ()
    artist_words =[w for w in artist_name_lower .split ()if len (w )>=2 ]
    song_title_words =[w for w in re .sub (r"[^a-z0-9 ]","",song_title_lower ).split ()if len (w )>=2 ]
    def _filter (info_dict ):
        title =info_dict .get ("title","").lower ()
        uploader =info_dict .get ("uploader","").lower ()
        channel =info_dict .get ("channel",uploader ).lower ()
        duration =info_dict .get ("duration")or 0 
        is_verified =bool (info_dict .get ("channel_is_verified",False ))
        view_count =info_dict .get ("view_count")or 0 
        if duration :
            if is_original and (duration >600 or duration <90 ):
                return f"Duration {duration}s outside range for originals (1.5–10 min)"
            elif not is_original and (duration >600 or duration <60 ):
                return f"Duration {duration}s outside range for covers (1–10 min)"
        bad_words =[
        "karaoke",
        "nightcore",
        "slowed",
        "reverb",
        "8d",
        "bass boosted",
        "tutorial",
        "synthesia",
        "reaction",
        "remix",
        "mashup",
        "acapella",
        "a cappella",
        "vocals only",
        "extended",
        "ai cover",
        "8 bit",
        "instrumental",
        "drum cover",
        "drum only",
        "play along",
        "play-along",
        "playalong",
        "backing track",
        "bass cover",
        "saxophone cover",
        "sax cover",
        "trumpet cover",
        "flute cover",
        "violin cover",
        "cello cover",
        "trombone cover",
        "ukulele cover",
        "piano only",
        "guitar playthrough",
        ]
        for word in bad_words :
            if word in title and word not in song_title_lower :
                return f'Rejected: title contains "{word}"'
        if re .search (r"\bai\b",title )and not re .search (r"\bai\b",song_title_lower ):
            return 'Rejected: title contains isolated word "AI" (likely AI generated)'
        live_patterns =[
        "(live)",
        "[live]",
        "live at ",
        "live from ",
        "live version",
        "live session",
        "live recording",
        "live performance",
        "live concert",
        "- live",
        "| live",
        ]
        for pattern in live_patterns :
            if pattern in title and pattern not in song_title_lower :
                return f'Rejected: live recording pattern "{pattern}"'
        if is_original :
            channel_matches =bool (artist_words )and any (
            w in channel or w in uploader for w in artist_words 
            )
            title_plain =re .sub (r"[^a-z0-9 ]","",title )
            song_in_title =not song_title_words or any (w in title_plain for w in song_title_words )
            if not song_in_title :
                return f'Rejected: video title does not contain any words from song "{song_title}"'
            if is_verified and channel_matches :
                if view_count <10000 :
                    return f"Rejected: verified artist channel but only {view_count} views — likely wrong video"
                return None 
            if channel_matches :
                if view_count <50000 :
                    return (
                    f"Rejected: channel matches artist but only {view_count} views"
                    )
                return None 
            if view_count <500000 :
                return (
                f'Rejected: unrecognised channel "{uploader}" '
                f"(not verified / not artist) with only {view_count} views"
                )
        else :
            if view_count <500 :
                return f"Rejected: cover has only {view_count} views (too low)"
            blocked_uploaders ={"vevo","topic","genius","lyrics"}
            if any (b in uploader or b in channel for b in blocked_uploaders ):
                return f'Rejected: uploader "{uploader}" is a blocked channel type (interviews/lyrics/official)'
            if artist_words and any (
            w in uploader or w in channel for w in artist_words 
            ):
                return (
                f"Rejected: uploader '{uploader}' appears to be the original artist"
                )
            if "cover"not in title :
                return 'Rejected: title does not contain "cover" — likely not a cover'
            if artist_words and not any (w in title for w in artist_words ):
                return f"Rejected: cover title does not mention original artist '{artist_name}'"
            if song_title_lower and song_title_lower not in title :
                core_title =re .sub (r"\(.*?\)","",song_title_lower ).strip ()
                if core_title and core_title not in title :
                    return f'Rejected: cover title does not contain exact song title "{core_title}"'
            return None 
    return _filter 
def append_to_metadata (song_id ,title ,orig_artist ,orig_url ,chosen_covers ):
    with CSV_LOCK :
        file_exists =os .path .exists (DATASET_CSV_PATH )
        with open (DATASET_CSV_PATH ,"a",newline ="",encoding ="utf-8")as f :
            writer =csv .writer (f )
            if not file_exists :
                writer .writerow (
                ["song_id","track_id","title","artist","version","url"]
                )
            writer .writerow (
            [song_id ,f"{song_id}_1",title ,orig_artist ,"original",orig_url ]
            )
            for i ,cover in enumerate (chosen_covers ):
                ver_name ="cover"if i ==0 else f"cover_{i+1}"
                writer .writerow (
                [song_id ,f"{song_id}_{i+2}",title ,cover ['artist'],ver_name ,cover ['url']]
                )
            f .flush ()
            os .fsync (f .fileno ())
def append_to_exhausted (artist ,title ,reason ="No suitable cover found"):
    with CSV_LOCK :
        file_exists =os .path .exists (EXHAUSTED_CSV_PATH )
        try :
            with open (EXHAUSTED_CSV_PATH ,"a",newline ="",encoding ="utf-8")as f :
                writer =csv .writer (f )
                if not file_exists :
                    writer .writerow (["artist","title","reason"])
                writer .writerow ([artist ,title ,reason ])
                f .flush ()
                os .fsync (f .fileno ())
        except Exception as e :
            print (
            f"      ⚠️ Warning: Failed to write to blacklist for '{artist} - {title}': {e}"
            )
def check_audio_not_silent (path ,rms_threshold =0.001 ):
    try :
        try :
            y ,_ =librosa .load (path ,sr =22050 ,mono =True ,offset =15.0 ,duration =10.0 )
        except Exception :
            y ,_ =librosa .load (path ,sr =22050 ,mono =True ,duration =10.0 )

        if len (y )==0 :
            print (f"      🔇 Rejected: Audio file is empty or corrupted")
            return False 

        rms =float (np .sqrt (np .mean (y **2 )))
        db_fs =20 *np .log10 (rms +1e-9 )
        if db_fs <-40 or rms <rms_threshold :
            print (
            f"      🔇 Rejected: Audio is too quiet (RMS={rms:.5f}, ~{db_fs:.1f} dBFS)"
            )
            return False 
        peak_amp =np .max (np .abs (y ))
        if peak_amp >1.10 :
            print (
            f"      💥 Rejected: Audio is severely clipping/distorted (Peak={peak_amp:.4f})"
            )
            return False 
        zcr =float (np .mean (librosa .feature .zero_crossing_rate (y )))
        flatness =float (np .mean (librosa .feature .spectral_flatness (y =y )))
        if zcr >0.35 or flatness >0.20 :
            print (
            f"      📻 Rejected: Audio quality fundamentally poor (ZCR={zcr:.3f}, Flatness={flatness:.3f})"
            )
            return False 
        return True 
    except Exception as e :
        print (f"      ⚠️ Could not validate audio volume/quality for {path}: {e}")
        return True 
def compute_chroma_fingerprint (path ):
    try :
        y ,sr =librosa .load (path ,sr =22050 ,mono =True )
        chroma =librosa .feature .chroma_cqt (y =y ,sr =sr )
        fp =np .concatenate ([chroma .mean (axis =1 ),chroma .std (axis =1 )])
        return fp 
    except Exception as e :
        print (f"      ⚠️ Fingerprint compute failed for {path}: {e}")
        return None 
def fingerprints_match (fp1 ,fp2 ,threshold =0.997 ):
    if fp1 is None or fp2 is None :
        return False 
    norm1 =np .linalg .norm (fp1 )
    norm2 =np .linalg .norm (fp2 )
    if norm1 ==0 or norm2 ==0 :
        return False 
    cosine_sim =float (np .dot (fp1 ,fp2 )/(norm1 *norm2 ))
    return cosine_sim >=threshold 
def verify_video_title (video_title ,artist ,song_title ,uploader_name =""):
    title_lower =video_title .lower ()
    uploader_lower =uploader_name .lower ()
    artist_words =[w for w in artist .lower ().split ()if len (w )>=2 ]
    song_words =[w for w in song_title .lower ().split ()if len (w )>=2 ]
    song_ok =not song_words or any (w in title_lower for w in song_words )
    if not song_ok :
        return False 
    artist_in_title =bool (artist_words )and any (w in title_lower for w in artist_words )
    artist_in_channel =bool (artist_words )and any (
    w in uploader_lower for w in artist_words 
    )
    is_official_channel =(
    "vevo"in uploader_lower 
    or "topic"in uploader_lower 
    or "official"in uploader_lower 
    )
    return artist_in_title or artist_in_channel or is_official_channel 
def compute_segment_stdev (orig_path ,cover_path ,n_windows =20 ):
    try :
        y_orig ,sr =librosa .load (orig_path ,sr =22050 ,mono =True )
        y_cover ,_ =librosa .load (cover_path ,sr =22050 ,mono =True )
        min_len =min (len (y_orig ),len (y_cover ))
        y_orig =y_orig [:min_len ]
        y_cover =y_cover [:min_len ]
        hop =min_len //n_windows 
        if hop ==0 :
            return None ,None 
        sims =[]
        for i in range (n_windows ):
            s ,e =i *hop ,(i +1 )*hop 
            seg_o =y_orig [s :e ]
            seg_c =y_cover [s :e ]
            chroma_o =librosa .feature .chroma_cqt (y =seg_o ,sr =sr ).mean (axis =1 )
            chroma_c =librosa .feature .chroma_cqt (y =seg_c ,sr =sr ).mean (axis =1 )
            norm_o =np .linalg .norm (chroma_o )
            norm_c =np .linalg .norm (chroma_c )
            if norm_o >0 and norm_c >0 :
                sims .append (float (np .dot (chroma_o ,chroma_c )/(norm_o *norm_c )))
        if not sims :
            return None ,None 
        return float (np .mean (sims )),float (np .std (sims ))
    except Exception as e :
        print (f"      ⚠️ Segment stdev computation failed: {e}")
        return None ,None 
def _process_candidate (args ):
    if SHUTDOWN_EVENT .is_set ():
        return 
    artist ,title =args 
    clean_title =str (sanitize_spotify_title (title ))
    safe_folder_name =re .sub (r'[<>:"/\\|?*]','',clean_title ).strip ()
    temp_folder_name =f"temp_{uuid.uuid4().hex[:6]}_{safe_folder_name[:15]}"
    target_dir =os .path .join (DATASET_DIR ,temp_folder_name )
    print (f"[PENDING] Processing: {artist} - {title}")
    os .makedirs (target_dir ,exist_ok =True )
    try :
        safe_artist =_sanitize_query (artist )
        safe_title =_sanitize_query (clean_title )
        orig_path =os .path .join (target_dir ,"original.wav")
        orig_query =f"{safe_artist} {safe_title} official audio"
        print ("   -> Downloading Original...")
        orig_success ,orig_url ,orig_uploader ,orig_video_title =download_audio (
        orig_query ,
        orig_path ,
        is_original =True ,
        artist_name =artist ,
        song_title =title ,
        )
        if not orig_success :
            print ("   ⏩ Skipping... Original failed.\n")
            append_to_exhausted (artist ,title ,reason ="Could not find original song on YouTube")
            return 
        if not check_audio_not_silent (orig_path ):
            print ("   ⏩ Skipping... Original audio is silent/corrupt.\n")
            append_to_exhausted (artist ,title ,reason ="Downloaded original was silent/corrupt")
            return 
        if not verify_video_title (orig_video_title ,artist ,title ,orig_uploader ):
            print (f"   ⚠️ QG#4 FAIL: Original track '{orig_video_title}' by '{orig_uploader}' doesn't match '{artist} - {title}'. Blacklisting.\n")
            append_to_exhausted (artist ,title ,reason =f"Original video mismatch: '{orig_video_title}' by '{orig_uploader}'")
            return 
        print ("   🔑 Computing original audio fingerprint...")
        orig_fingerprint =compute_chroma_fingerprint (orig_path )
        orig_dur =librosa .get_duration (path =orig_path )
        cover_query =f"{safe_artist} {safe_title} cover"
        print (f"   -> Searching covers: '{cover_query}'")
        _cover_filter =get_yt_dlp_filter (is_original =False ,artist_name =artist ,song_title =title )
        seen_ids ={_youtube_video_id (orig_url )}
        cover_candidates =[]
        try :
            api_results =search_youtube_api_with_metadata (cover_query ,max_results =30 ,category_id =10 )
            if not api_results :
                print ("      ⚠️ API returned 0 covers. Moving to next track...")
            for entry in api_results :
                reason =_cover_filter (entry )
                vid_id =entry .get ("id","")
                if reason is None and vid_id not in seen_ids :
                    cover_candidates .append (entry )
        except Exception as e :
            if isinstance (e ,QuotaExceededError ):
                raise 
            print (f"   ❌ Cover search failed: {e}")
        cover_candidates .sort (key =lambda e :e .get ("view_count")or 0 ,reverse =True )
        print (f"   🎵 {len(cover_candidates)} candidates after filter. Taking first valid one...")
        chosen_covers =[]
        seen_channel_ids =set ()
        max_covers =3 
        for idx ,candidate in enumerate (cover_candidates ):
            if SHUTDOWN_EVENT .is_set ():
                return 
            if len (chosen_covers )>=max_covers :
                break 
            cand_url =candidate .get ("webpage_url","")
            cand_artist =candidate .get ("uploader","Unknown")
            cand_yttitle =candidate .get ("title","?")
            cand_views =candidate .get ("view_count")or 0 
            cand_channel_id =candidate .get ("channel_id")or candidate .get ("uploader_id")or ""
            if cand_channel_id and cand_channel_id in seen_channel_ids :
                print (f"   -> Skipping '{cand_yttitle}' (same channel as existing cover).")
                continue 
            current_cover_path =os .path .join (target_dir ,f"temp_cover_{idx+1}.wav")
            print (f"   -> [{idx+1}/{len(cover_candidates)}] '{cand_yttitle}' | {cand_artist} | {cand_views:,} views")
            if not _download_url (cand_url ,current_cover_path ):
                print ("      ❌ Download failed. Trying next candidate...")
                continue 
            seen_ids .add (_youtube_video_id (cand_url ))
            if not check_audio_not_silent (current_cover_path ):
                safe_remove_file (current_cover_path )
                continue 
            cover_fingerprint =compute_chroma_fingerprint (current_cover_path )
            if fingerprints_match (orig_fingerprint ,cover_fingerprint ):
                print ("      🔑 Fingerprint matches original — same recording. Skipping...")
                safe_remove_file (current_cover_path )
                continue 
            cov_dur =librosa .get_duration (path =current_cover_path )
            dur_diff_ratio =abs (orig_dur -cov_dur )/max (1 ,orig_dur )
            if dur_diff_ratio >0.15 :
                print (f"      ⚠️ Duration gap too large ({orig_dur:.1f}s vs {cov_dur:.1f}s). Skipping.")
                safe_remove_file (current_cover_path )
                continue 
            ver_name ="cover.wav"if len (chosen_covers )==0 else f"cover_{len(chosen_covers)+1}.wav"
            final_cover_path =os .path .join (target_dir ,ver_name )
            rename_ok =False 
            for rename_attempt in range (3 ):
                try :
                    os .rename (current_cover_path ,final_cover_path )
                    rename_ok =True 
                    break 
                except Exception as rename_err :
                    if rename_attempt <2 :
                        time .sleep (1.5 )
                    else :
                        print (f"      ❌ Could not rename cover {ver_name}: {rename_err}")
            if rename_ok :
                chosen_covers .append ({
                'path':final_cover_path ,
                'url':cand_url ,
                'artist':cand_artist ,
                'title':cand_yttitle 
                })
                if cand_channel_id :
                    seen_channel_ids .add (cand_channel_id )
                print (f"      ✅ Accepted! Saved as {ver_name}: '{cand_yttitle}'")
                global channel_counts 
                with DICT_LOCK :
                    channel_counts [cand_artist ]=channel_counts .get (cand_artist ,0 )+1 
            else :
                safe_remove_file (current_cover_path )
        if not chosen_covers :
            print (f"   ☠️ Failed to find any good covers for '{title}'. Blacklisting.")
            safe_delete_folder (target_dir )
            append_to_exhausted (artist ,title ,reason ="No suitable cover found")
            return 
        print (f"   ✅ Found {len(chosen_covers)} good covers for {title}! Saving to CSV...")
        for f in os .listdir (target_dir ):
            if f not in ("original.wav","cover.wav","cover_2.wav","cover_3.wav"):
                safe_remove_file (os .path .join (target_dir ,f ))
        final_folder_num =get_realtime_folder_id ()
        final_folder_name =f"{final_folder_num:03d}"
        final_dir =os .path .join (DATASET_DIR ,final_folder_name )
        if os .path .exists (final_dir ):
            print (f"   ⚠️ Folder {final_folder_name} already exists on disk — skipping rename to avoid overwrite.")
            with REALTIME_ID_LOCK :
                realtime_gap_queue .insert (0 ,final_folder_num )
            return 
        rename_success =False 
        import time 
        for attempt in range (5 ):
            try :
                os .rename (target_dir ,final_dir )
                rename_success =True 
                break 
            except Exception as e :
                time .sleep (1.5 )
        if not rename_success :
            print (f"   ❌ FATAL: Could not rename {target_dir} to {final_dir} after 5 attempts.")
            with REALTIME_ID_LOCK :
                realtime_gap_queue .insert (0 ,final_folder_num )
            return 
        print (f"   🏆 FINISHED: Successfully wrote {artist} - {title} to folder [{final_folder_name}]")
        append_to_metadata (final_folder_name ,title ,artist ,orig_url ,chosen_covers )
    finally :
        if os .path .exists (target_dir ):
            safe_delete_folder (target_dir )
            print (f"   🧹 Swept temp folder for {artist} - {title}")
def remove_dataset_duplicates ():
    if not os .path .exists (DATASET_CSV_PATH ):return 
    seen_songs =set ()
    rows_to_keep =[]
    removed_ids =set ()
    cleaned =False 
    with open (DATASET_CSV_PATH ,"r",encoding ="utf-8")as f :
        reader =csv .reader (f )
        header =next (reader ,None )
        if header :rows_to_keep .append (header )
        for row in reader :
            if len (row )>=5 :
                s_id ,_ ,title ,artist ,version =row [:5 ]
                if version .lower ()=="original":
                    _t =re .sub (r"[^a-z0-9 ]","",title .lower ().strip ())
                    _a =re .sub (r"[^a-z0-9 ]","",artist .lower ().strip ())
                    song_key =(_a ,_t )
                    if song_key in seen_songs or s_id in removed_ids :
                        removed_ids .add (s_id )
                        cleaned =True 
                        print (f"   🗑️ Auto-delete duplicate found: [{s_id}] {artist} - {title}")
                    else :
                        seen_songs .add (song_key )
                        rows_to_keep .append (row )
                else :
                    if s_id in removed_ids :
                        cleaned =True 
                    else :
                        rows_to_keep .append (row )
            else :
                rows_to_keep .append (row )
    if cleaned :
        with CSV_LOCK :
            with open (DATASET_CSV_PATH ,"w",newline ="",encoding ="utf-8")as f :
                writer =csv .writer (f )
                writer .writerows (rows_to_keep )
        for s_id in removed_ids :
            try :
                folder_name =f"{int(s_id):03d}"
                full_path =os .path .join (DATASET_DIR ,folder_name )
                feature_path =os .path .join ("features",folder_name )
                safe_delete_folder (full_path ,retries =1 ,delay =0.5 )
                safe_delete_folder (feature_path ,retries =1 ,delay =0.5 )
            except ValueError :
                pass 
        print (f"   ✨ Successfully deleted {len(removed_ids)} duplicate dataset and feature folder(s) to free up space.")
def main ():
    print ("="*50 )
    print ("YT-DLP DATASET SCRAPER (Spotify Final CSV) - THREADED")
    print ("="*50 )
    os .makedirs (DATASET_DIR ,exist_ok =True )
    cleanup_temp_folders ()
    remove_dataset_duplicates ()
    if not os .path .exists (SPOTIFY_CSV_PATH ):
        print (f"❌ Could not find {SPOTIFY_CSV_PATH}.")
        sys .exit (1 )
    existing_songs ,next_new_id ,gap_ids ,orphan_ids =get_existing_songs ()
    print (f"➡️ Next available folder ID: {max(next_new_id + 1, 1):03d}")
    if orphan_ids :
        print ("❌ HALTED — Orphan folder(s) detected. Please fix manually.")
        sys .exit (1 )
    if gap_ids :
        print (f"🕳️ Found {len(gap_ids)} gap(s) to fill first.")
    candidates =get_candidate_songs (existing_songs )
    print (f"🎯 Selected {len(candidates)} brand new track pairs to scrape.\n")
    job_args =[]
    init_realtime_id_pool (gap_ids ,next_new_id +1 )
    for artist ,title in candidates :
        job_args .append ((artist ,title ))
    global channel_counts 
    channel_counts ={}
    workers =4 
    print (f"🚀 Launching thread pool with {workers} concurrent downloading workers...\n")
    executor =ThreadPoolExecutor (max_workers =workers )
    job_futures ={}
    for arg in job_args :
        future =executor .submit (_process_candidate ,arg )
        job_futures [future ]=arg 
    import concurrent .futures 
    try :
        pending =set (job_futures .keys ())
        while pending :
            if SHUTDOWN_EVENT .is_set ():
                break 
            wait_res =concurrent .futures .wait (
            pending ,timeout =1.0 ,return_when =concurrent .futures .FIRST_COMPLETED 
            )
            done =set (wait_res .done )
            pending =set (wait_res .not_done )
            for future in done :
                try :
                    future .result ()
                except QuotaExceededError :
                    print ("\n🛑 Quota Exceeded Error caught at top level. Shutting down scraper gracefully...")
                    SHUTDOWN_EVENT .set ()
                    cleanup_temp_folders ()
                    executor .shutdown (wait =False ,cancel_futures =True )
                    sys .exit (1 )
                except Exception as e :
                    arg =job_futures [future ]
                    print (f"      ❌ Worker crashed for {arg}: {e}")
            free_bytes =shutil .disk_usage (DATASET_DIR ).free 
            if free_bytes <5 *1024 **3 :
                free_gb =free_bytes /(1024 **3 )
                print (f"\n🚨 DISK SPACE CRITICAL: Only {free_gb:.1f}GB remaining on drive!")
                print ("⛔ Stopping scraper automatically to prevent data corruption.")
                SHUTDOWN_EVENT .set ()
                break 
    except KeyboardInterrupt :
        print ("\n\n⛔ Ctrl+C received — force-stopping all workers...")
        SHUTDOWN_EVENT .set ()
        cleanup_temp_folders ()
        executor .shutdown (wait =False ,cancel_futures =True )
        print ("⛔ Exiting immediately.")
        os ._exit (1 )
    executor .shutdown (wait =True )
    print ("🎉 Scraping batch complete!")
    print (f"📊 Channel coverage this run: { {k: v for k, v in sorted(channel_counts.items(), key=lambda x: -x[1])} }")
if __name__ =="__main__":
    try :
        main ()
    except KeyboardInterrupt :
        print ("\n\n⛔ Interrupted by user. Exiting.")
        SHUTDOWN_EVENT .set ()
        try :
            cleanup_temp_folders ()
        except NameError :
            pass 
        os ._exit (1 )

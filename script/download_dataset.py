import yt_dlp
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
import time
import random

def normalize_text(text):
    if pd.isna(text):
        return ''
    return ' '.join(str(text).lower().strip().split())

def check_duplicates(csv_file='dataset.csv'):
    df = pd.read_csv(csv_file)
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce')
    df = df.dropna(subset=['song_id']).copy()
    df['song_id'] = df['song_id'].astype(int)
    print('=' * 70)
    print('🔍 CHECKING FOR DUPLICATES (Title + Original Artist)')
    print('    Rule: same title + same ORIGINAL artist = duplicate')
    print('          same title + different original artist = different song')
    print('=' * 70)
    song_identity = {}
    for (_, row) in df.iterrows():
        sid = str(row['song_id']).zfill(3)
        ver = str(row['version']).strip().lower()
        if ver == 'original':
            song_identity[sid] = (normalize_text(row['title']), normalize_text(row['artist']))
    identity_to_ids = defaultdict(list)
    for (sid, identity) in song_identity.items():
        identity_to_ids[identity].append(sid)
    duplicate_list = []
    for ((title_norm, artist_norm), song_ids) in identity_to_ids.items():
        if len(song_ids) > 1:
            duplicate_list.append({'title': title_norm, 'artist': artist_norm, 'song_ids': sorted(song_ids)})
    if duplicate_list:
        print(f'\n⚠️  FOUND {len(duplicate_list)} TRUE DUPLICATES:\n')
        for dup in duplicate_list:
            ids_str = ', '.join((f'song_id={s}' for s in dup['song_ids']))
            print(f'''   ❌ "{dup['title']}" originally by "{dup['artist']}"''')
            print(f'      Appears as: {ids_str}')
            print()
    else:
        print('\n✅ NO DUPLICATES FOUND! Dataset is clean.\n')
    print('=' * 70)
    return duplicate_list
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def _sweep_incomplete_downloads(output_dir):
    import shutil
    swept = []
    if not os.path.exists(output_dir):
        return swept
    for folder in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, folder)
        if not os.path.isdir(path):
            continue
        has_orig = os.path.exists(os.path.join(path, 'original.wav'))
        has_any_cover = any((os.path.exists(os.path.join(path, f'{c}.wav')) for c in ['cover', 'cover_2', 'cover_3']))
        if not (has_orig and has_any_cover):
            shutil.rmtree(path, ignore_errors=True)
            swept.append(folder)
    return swept

def download_dataset(csv_file='dataset.csv', output_dir='dataset', start_song=None, end_song=None, max_workers=12):
    if duplicates:
        print('\n⚠️  WARNING: Duplicates detected!')
        response = input('Continue download anyway? (y/n): ').strip().lower()
        if response != 'y':
            print('❌ Download cancelled. Please fix duplicates first.')
            return
    df = pd.read_csv(csv_file)
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce')
    df = df.dropna(subset=['song_id']).copy()
    df['song_id'] = df['song_id'].astype(int)
    if start_song is not None or end_song is not None:
        try:
            start_int = int(start_song) if start_song else None
            end_int = int(end_song) if end_song else None
        except ValueError:
            print('❌ --start-song / --end-song must be integers')
            return
        if start_int is not None:
            df = df[df['song_id'] >= start_int]
        if end_int is not None:
            df = df[df['song_id'] <= end_int]
        print(f"\n📌 Range filter: song_id {start_song or 'first'} → {end_song or 'last'}")
        print(f'   Tracks to download: {len(df)}')
    (delay_min, delay_max) = (0.2, 0.8)
    n_fragments = 8
    sock_timeout = 30
    import shutil as _shutil
    _use_aria2c = _shutil.which('aria2c') is not None
    _engine = 'aria2c' if _use_aria2c else f'native ({n_fragments} frags)'
    print('\n' + '=' * 70)
    print(f'📥 STARTING DOWNLOAD ({max_workers} threads)')
    print('=' * 70)
    print(f'📊 Total tracks : {len(df)}')
    print(f"📊 Total songs  : {df['song_id'].nunique()}")
    print(f'📁 Output dir   : {output_dir}/')
    print(f'🚀 Engine: {_engine} | Threads: {max_workers}')
    success_count = 0
    failed_list = []
    skipped_count = 0
    unavailable_song_ids: set = set()
    lock = threading.RLock()
    _rate_limit_until = [0.0]
    _stop_event = threading.Event()
    _bot_detect_count = [0]
    BOT_DETECT_STOP_AFTER = 3
    bot_skipped_song_ids: set = set()

    def _remove_song_immediately(sid_str: str, row_data: dict):
        import shutil as _shutil
        with lock:
            try:
                df_cur = pd.read_csv(csv_file)
                before = len(df_cur)
                df_cur['_sid'] = df_cur['song_id'].astype(str)
                df_removed = df_cur[df_cur['_sid'] == sid_str]
                df_cur = df_cur[df_cur['_sid'] != sid_str].drop(columns=['_sid'])
                if df_removed.empty:
                    return
                df_cur.to_csv(csv_file, index=False)
                tqdm.write(f'  🗑️  Song {sid_str} removed from dataset.csv ({before}→{len(df_cur)} rows)')
            except Exception as e:
                tqdm.write(f'  ⚠️  Could not clean up CSV for song {sid_str}: {e}')

    def process_row(row):
        nonlocal success_count, skipped_count, failed_list
        with lock:
            if str(row['song_id']) in unavailable_song_ids:
                return (True, None)
        song_id = str(row['song_id']).zfill(3)
        track_id = row['track_id']
        version = row['version']
        url = row['url']
        title = row['title']
        artist = row['artist']
        song_dir = f'{output_dir}/{song_id}'
        os.makedirs(song_dir, exist_ok=True)
        output_file = f'{song_dir}/{version}'
        output_path = f'{output_file}.wav'
        if os.path.exists(output_path):
            with lock:
                skipped_count += 1
            return (True, None)
        if pd.isna(url) or str(url).strip() == '':
            return (False, {'track_id': track_id, 'song_id': song_id, 'title': title, 'artist': artist, 'reason': 'No URL'})
        ydl_opts = {'format': 'bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '0'}], 'quiet': True, 'no_warnings': True, 'retries': 5, 'fragment_retries': 10, 'concurrent_fragment_downloads': n_fragments, 'http_chunk_size': 10 * 1024 * 1024, 'socket_timeout': sock_timeout, 'nocheckcertificate': True, 'extractor_args': {'youtube': {'player_client': ['android', 'ios', 'web']}}, 'outtmpl': f'{output_file}.%(ext)s'}
        if _use_aria2c:
            ydl_opts['external_downloader'] = 'aria2c'
            ydl_opts['external_downloader_args'] = {'default': ['-x', '16', '-s', '16', '-k', '1M', '--min-split-size=1M', '--quiet']}
        download_success = False
        last_error = None
        for attempt in range(3):
            if _stop_event.is_set():
                with lock:
                    skipped_count += 1
                return (True, None)
            wait = _rate_limit_until[0] - time.time()
            if wait > 0:
                _stop_event.wait(wait)
            if _stop_event.is_set():
                break
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                with lock:
                    success_count += 1
                download_success = True
                _stop_event.wait(random.uniform(delay_min, delay_max))
                break
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                if 'HTTP Error 429' in error_str:
                    backoff = 30 * 3 ** attempt
                    with lock:
                        _rate_limit_until[0] = max(_rate_limit_until[0], time.time() + backoff)
                    _stop_event.wait(backoff)
                elif any((kw in error_str for kw in ('Sign in to confirm', 'not a bot', 'confirm you are not a bot', 'Please sign in', 'bot check', 'detected unusual traffic'))):
                    with lock:
                        _bot_detect_count[0] += 1
                        n_bots = _bot_detect_count[0]
                        sid_str = str(row['song_id'])
                        if sid_str not in unavailable_song_ids and sid_str not in bot_skipped_song_ids:
                            bot_skipped_song_ids.add(sid_str)
                            unavailable_song_ids.add(sid_str)
                            tqdm.write(f'\n  🤖 Bot detected ({song_id}/{version}) — paired track skipped, CSV NOT modified')
                        if n_bots >= BOT_DETECT_STOP_AFTER and (not _stop_event.is_set()):
                            _stop_event.set()
                            tqdm.write(f"\n\n{'=' * 60}")
                            tqdm.write(f'  🚨 BOT DETECTED {n_bots}x — STOPPING ALL DOWNLOADS')
                            tqdm.write(f'  ⏸️  YouTube is actively blocking this session.')
                            tqdm.write(f'  ⏳ Wait 30-60 minutes then run again to resume.')
                            tqdm.write(f'  ✅ Progress is saved — no songs will be re-downloaded.')
                            tqdm.write(f"{'=' * 60}")
                    break
                _PERMANENT_ERRORS = ('unavailable', 'private', 'terminated', 'removed', 'deleted', 'no longer available', 'account associated', 'does not exist')
                if any((kw in error_str.lower() for kw in _PERMANENT_ERRORS)):
                    sid_str = str(row['song_id'])
                    tqdm.write(f'  ⚠️  Song {song_id} [{version}] PERMANENTLY UNAVAILABLE')
                    tqdm.write(f'     URL: {url}')
                    with lock:
                        if sid_str not in unavailable_song_ids:
                            unavailable_song_ids.add(sid_str)
                            _remove_song_immediately(sid_str, row)
                    break
                else:
                    tqdm.write(f'  ⚠️  [{song_id}/{version}] Attempt {attempt + 1}/3 failed: {error_str[:100]}')
                    _stop_event.wait(10 * (attempt + 1))
        if not download_success:
            if last_error is None and _stop_event.is_set():
                with lock:
                    skipped_count += 1
                return (True, None)
            sid_str = str(row['song_id'])
            is_bot_skip = sid_str in bot_skipped_song_ids
            for _leftover in os.listdir(song_dir) if os.path.exists(song_dir) else []:
                if _leftover.startswith(version) and (not _leftover.endswith('.wav')):
                    try:
                        os.remove(os.path.join(song_dir, _leftover))
                    except Exception:
                        pass
            if not is_bot_skip:
                err_short = (last_error or 'Unknown')[:200]
                tqdm.write(f'\n  ❌ FAILED ({song_id}/{version}): {title} — {artist}')
                tqdm.write(f'     Error : {err_short[:120]}')
                tqdm.write(f'     URL   : {url}')
                with lock:
                    if sid_str not in unavailable_song_ids:
                        unavailable_song_ids.add(sid_str)
                        tqdm.write(f'  🗑️  Removing from dataset.csv (all retries exhausted)...')
                        _remove_song_immediately(sid_str, row)
            return (False, {'track_id': track_id, 'song_id': song_id, 'title': title, 'artist': artist, 'url': url, 'error': (last_error or 'Unknown')[:200]})
        return (True, None)
    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, row): row for (_, row) in df.iterrows()}
            try:
                for future in tqdm(as_completed(futures), total=len(futures), desc='📥 Downloading'):
                    (success, failure_data) = future.result()
                    if not success and failure_data:
                        with lock:
                            failed_list.append(failure_data)
            except KeyboardInterrupt:
                interrupted = True
                _stop_event.set()
                print('\n\n🛑 Ctrl+C received — stopping threads...')
                for fut in futures:
                    fut.cancel()
    except KeyboardInterrupt:
        interrupted = True
        _stop_event.set()
    print('\n🧹 Sweeping incomplete song folders...')
    swept = _sweep_incomplete_downloads(output_dir)
    if swept:
        print(f"   Removed {len(swept)} incomplete folder(s): {', '.join(swept[:10])}{('...' if len(swept) > 10 else '')}")
    else:
        print('   Nothing to sweep — all folders are clean.')
    if interrupted:
        print('✅ Dataset directory is now clean. Resume with the same command.')
    print('\n' + '=' * 70)
    print('📊 DOWNLOAD SUMMARY')
    print('=' * 70)
    print(f'✅ Success:  {success_count}/{len(df)} ({success_count / len(df) * 100:.1f}%)')
    print(f'❌ Failed:   {len(failed_list)}/{len(df)} ({len(failed_list) / len(df) * 100:.1f}%)')
    print(f'⏭️  Skipped:  {skipped_count}/{len(df)} (already exist)')
    print('=' * 70)
    if failed_list:
        error_types = defaultdict(int)
        for item in failed_list:
            error = str(item.get('error', 'Unknown'))
            if 'Sign in' in error or 'bot' in error:
                error_types['Bot detection'] += 1
            elif 'unavailable' in error.lower():
                error_types['Unavailable'] += 1
            elif 'HTTP Error 429' in error:
                error_types['Rate limited'] += 1
            elif 'No URL' in error:
                error_types['No URL'] += 1
            else:
                error_types['Other'] += 1
        print('\n📊 Error types:')
        for (error_type, count) in error_types.items():
            print(f'   {error_type}: {count}')
    print('\n✨ Download complete!')
    return {'success': success_count, 'failed': len(failed_list), 'skipped': skipped_count}

def verify_dataset_structure(output_dir='dataset'):
    print('\n' + '=' * 70)
    print('🔍 VERIFYING DATASET')
    print('=' * 70)
    if not os.path.exists(output_dir):
        print(f'\n❌ Directory not found: {output_dir}')
        return None
    folders = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))])
    if not folders:
        print(f'\n❌ No folders in {output_dir}')
        return None
    print(f'\n📁 Checking {len(folders)} folders...\n')
    complete = []
    incomplete = []
    for song_id_str in folders:
        song_dir = os.path.join(output_dir, song_id_str)
        original = os.path.join(song_dir, 'original.wav')
        has_orig = os.path.exists(original)
        cover_versions = [c for c in ['cover', 'cover_2', 'cover_3'] if os.path.exists(os.path.join(song_dir, f'{c}.wav'))]
        has_any_cover = len(cover_versions) > 0
        info = {'song_id': song_id_str, 'has_original': has_orig, 'cover_versions': cover_versions, 'status': 'complete' if has_orig and has_any_cover else 'incomplete'}
        if has_orig and has_any_cover:
            complete.append(info)
        else:
            incomplete.append(info)
            missing = []
            if not has_orig:
                missing.append('original')
            if not has_any_cover:
                missing.append('cover (any version)')
            print(f"⚠️  {song_id_str}: Missing {', '.join(missing)}")
    print('\n' + '=' * 70)
    print('📊 VERIFICATION SUMMARY')
    print('=' * 70)
    print(f'✅ Complete:   {len(complete)}/{len(folders)}')
    print(f'⚠️  Incomplete: {len(incomplete)}/{len(folders)}')
    print('=' * 70)
    if incomplete:
        print(f'\n⚠️  Found {len(incomplete)} incomplete folders:')
        for res in incomplete:
            missing = []
            if not res['has_original']:
                missing.append('original')
            if not res['cover_versions']:
                missing.append('cover')
            print(f"   ❌ {res['song_id']}: Missing {', '.join(missing)}")
    return {'total': len(folders), 'complete': len(complete), 'incomplete': len(incomplete), 'incomplete_list': incomplete}

def download_single(url, output_path, quiet=False):
    base_path = os.path.splitext(output_path)[0]
    ydl_opts = {'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '0'}], 'quiet': quiet, 'no_warnings': quiet, 'retries': 5, 'socket_timeout': 120, 'nocheckcertificate': True, 'extractor_args': {'youtube': {'player_client': ['android', 'ios', 'web']}}, 'outtmpl': f'{base_path}.%(ext)s'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if os.path.exists(output_path):
            print(f'✅ Saved: {output_path}')
            return True
        for ext in ['wav', 'mp3', 'opus', 'm4a', 'webm']:
            alt = f'{base_path}.{ext}'
            if os.path.exists(alt):
                print(f'✅ Saved: {alt}')
                return True
        print('❌ File not found after download')
        return False
    except Exception as e:
        print(f'❌ Download error: {e}')
        return False

def interactive_single_download():
    print('\n' + '=' * 70)
    print('🎵 SINGLE SONG DOWNLOADER (same quality as training data)')
    print('=' * 70)
    while True:
        print('\nOptions:')
        print('  1. Save to dataset/ folder (original/cover pair)')
        print('  2. Save to custom path   (for query testing)')
        print('  q. Quit')
        choice = input('\n🔢 Choice: ').strip().lower()
        if choice == 'q':
            break
        url = input('📎 YouTube URL: ').strip()
        if not url:
            print('❌ No URL given')
            continue
        if choice == '1':
            song_id = input('📁 Song ID (e.g. 001): ').strip().zfill(3)
            version = input('🎵 Version (original/cover): ').strip().lower()
            if version not in ['original', 'cover']:
                print("❌ Must be 'original' or 'cover'")
                continue
            song_dir = os.path.join('dataset', song_id)
            os.makedirs(song_dir, exist_ok=True)
            output_path = os.path.join(song_dir, f'{version}.wav')
        elif choice == '2':
            filename = input('💾 Output filename (e.g. test_song.wav): ').strip()
            if not filename.endswith('.wav'):
                filename += '.wav'
            output_path = filename
        else:
            print('❌ Invalid choice')
            continue
        if os.path.exists(output_path):
            ow = input(f'⚠️  {output_path} exists. Overwrite? (y/n): ').strip().lower()
            if ow != 'y':
                print('⏭️  Skipped')
                continue
        print(f'\n📥 Downloading → {output_path}')
        download_single(url, output_path)
        cont = input('\n▶️  Download another? (y/n): ').strip().lower()
        if cont != 'y':
            break
    print('\n✨ Done!')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download dataset from YouTube')
    parser.add_argument('--start-song', type=str, default=None, help='Download only songs with song_id >= this value (e.g. 1 or 4001)')
    parser.add_argument('--end-song', type=str, default=None, help='Download only songs with song_id <= this value (e.g. 4000 or 5000)')
    parser.add_argument('--threads', type=int, default=12, help='Override number of download threads (default: 12)')
    parser.add_argument('--single', action='store_true', help='Interactive single-song download')
    args = parser.parse_args()
    if args.single:
        interactive_single_download()
    else:
        duplicates = check_duplicates('dataset.csv')
        if duplicates:
            print('\n⚠️  WARNING: True duplicates detected!')
            response = input('Continue despite duplicates? (y/n): ').strip().lower()
            if response != 'y':
                print('❌ Cancelled.')
                exit(0)
        verify = verify_dataset_structure('dataset')
        df_full = pd.read_csv('dataset.csv')
        df_full['song_id'] = pd.to_numeric(df_full['song_id'], errors='coerce')
        df_full = df_full.dropna(subset=['song_id']).copy()
        df_full['song_id'] = df_full['song_id'].astype(int)
        df_range = df_full.copy()
        if args.start_song:
            df_range = df_range[df_range['song_id'] >= int(args.start_song)]
        if args.end_song:
            df_range = df_range[df_range['song_id'] <= int(args.end_song)]
        total_songs_range = df_range['song_id'].nunique()
        tracks_to_download = len(df_range)
        avg_time = 12
        pause_total = tracks_to_download // 10 * 60
        total_min = (tracks_to_download * avg_time + pause_total) / 60
        print(f"\n{'─' * 70}")
        if args.start_song or args.end_song:
            print(f"  Range            : song {args.start_song or 'first'} → {args.end_song or 'last'}")
        print(f'  Songs in range   : {total_songs_range:>5}')
        print(f'  Tracks to fetch  : {tracks_to_download:>5}')
        print(f'  Estimated time   : ~{total_min:.0f} min ({total_min / 60:.1f} hrs)')
        print(f"{'─' * 70}")
        response = input('\n🚀 Start download? (y/n): ').strip().lower()
        if response == 'y':
            download_dataset('dataset.csv', 'dataset', start_song=args.start_song, end_song=args.end_song, max_workers=args.threads)
            print('\n' + '=' * 70)
            print('🔍 FINAL VERIFICATION')
            print('=' * 70)
            verify_dataset_structure('dataset')
            print('\n🎉 Done!')
        else:
            print('❌ Download cancelled.')
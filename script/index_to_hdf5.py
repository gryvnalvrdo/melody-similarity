import os
import shutil
import tempfile
import numpy as np
from tqdm import tqdm
import gc
try:
    import h5py
except ImportError:
    print('❌ Critical: Please install h5py to create the database (pip install h5py)')
    exit(1)
try:
    from config import FEATURES_DIR, HDF5_DATASET_PATH
except Exception:
    try:
        from script.config import FEATURES_DIR, HDF5_DATASET_PATH
    except Exception:
        import os
        _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        FEATURES_DIR = os.path.join(_BASE, 'features')
        HDF5_DATASET_PATH = os.path.join(_BASE, 'dataset.h5')
_STOP = False

def create_hdf5_database(features_dir=FEATURES_DIR, out_path=HDF5_DATASET_PATH, cleanup_after_pack: bool=False):
    progress_path = out_path + '.progress'
    valid_songs = sorted([f for f in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, f)) and any((file.endswith('.npy') for file in os.listdir(os.path.join(features_dir, f))))])
    print(f'Found {len(valid_songs)} valid songs.')
    completed_ids: set = set()
    if os.path.exists(progress_path):
        with open(progress_path, 'r', encoding='utf-8') as pf:
            completed_ids = set((line.strip() for line in pf if line.strip()))
        remaining = [s for s in valid_songs if s not in completed_ids]
        print(f'⏩ Resuming: {len(completed_ids)} already done, {len(remaining)} remaining.')
    else:
        remaining = valid_songs
        if os.path.exists(out_path):
            resp = input(f'⚠️  {os.path.basename(out_path)} already exists. Overwrite? (y/n): ')
            if resp.lower() != 'y':
                print('Aborting.')
                return
            os.remove(out_path)
    if not remaining:
        print('✅ All songs already written. Nothing to do.')
        _finalize(out_path, progress_path)
        return
    h5_mode = 'a' if completed_ids else 'w'
    print(f'🗜️  Writing HDF5 directly to: {out_path}')
    print(f'   (Ctrl+C safe — progress saved after every song)\n')
    interrupted = False
    try:
        h5f = h5py.File(out_path, h5_mode)
        pf = open(progress_path, 'a', encoding='utf-8')
        for (i, song_id) in enumerate(tqdm(remaining, desc='Writing to Database')):
            if _STOP:
                interrupted = True
                break
            if i > 0 and i % 10 == 0:
                h5f.close()
                import gc
                gc.collect()
                h5f = h5py.File(out_path, 'a')
            song_dir = os.path.join(features_dir, song_id)
            group = h5f.require_group(song_id)
            song_ok = False
            for version in ['original', 'cover', 'cover_2', 'cover_3']:
                npy_path = os.path.join(song_dir, f'{version}.npy')
                if not os.path.exists(npy_path):
                    continue
                try:
                    data = np.load(npy_path, allow_pickle=True).item()
                    cqt = data.get('harmonic_cqt')
                    if cqt is None:
                        cqt = data.get('cqt_features')
                    if cqt is None:
                        continue
                    cqt = cqt.astype(np.float16)
                    (n_segs, n_bins, n_frames) = cqt.shape
                    valid_mask = (cqt.mean(axis=(1, 2)) > 0.0001).astype(np.uint8)
                    chunk_shape = (1, n_bins, n_frames)
                    dset = group.create_dataset(version, data=cqt, compression='lzf', chunks=chunk_shape)
                    dset.attrs['shift_to_c'] = data.get('shift_to_c', 0)
                    dset.attrs['global_key'] = data.get('global_key', 'Unknown')
                    dset.attrs['valid_mask'] = valid_mask
                    song_ok = True
                    del data, cqt, valid_mask, chunk_shape
                    dset = None
                except Exception as e:
                    print(f'\n  Error packing {song_id}/{version}: {e}')
            if not song_ok and song_id in h5f:
                del h5f[song_id]
            if song_ok:
                h5f.flush()
                pf.write(song_id + '\n')
                pf.flush()
                if cleanup_after_pack:
                    shutil.rmtree(song_dir, ignore_errors=True)
    except KeyboardInterrupt:
        interrupted = True
        print('\n\n🛑 Ctrl+C — stopping. Progress saved. Run again to resume.')
    finally:
        try:
            h5f.close()
        except:
            pass
        try:
            pf.close()
        except:
            pass
    if not interrupted:
        _finalize(out_path, progress_path)
    else:
        size_mb = os.path.getsize(out_path) / (1024 * 1024) if os.path.exists(out_path) else 0
        with open(progress_path, encoding='utf-8') as _pf_r:
            done_now = len(completed_ids) + len([l for l in _pf_r if l.strip()])
        print(f'   Saved so far: ~{size_mb:.0f} MB — run again with same command to resume.')

def _finalize(out_path, progress_path):
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'\n✅ HDF5 Database ready: {out_path}')
    print(f'   Size: {size_mb:.1f} MB')
    if os.path.exists(progress_path):
        os.remove(progress_path)
        print('   Progress checkpoint removed (clean run).')
    print('PyTorch will now stream from this database automatically.')

def sync_hdf5_database(features_dir=FEATURES_DIR, out_path=HDF5_DATASET_PATH):
    if not os.path.exists(out_path):
        print(f'❌ No database found at {out_path}. Run create_hdf5_database() first.')
        return
    live_song_ids = set((f for f in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, f)) and f != '__pycache__'))
    stale_keys = []
    with h5py.File(out_path, 'r') as h5f:
        for key in h5f.keys():
            if key not in live_song_ids:
                stale_keys.append(key)
    if not stale_keys:
        print('✅ HDF5 database is perfectly in sync. No stale keys found.')
        return
    print(f'🔍 Found {len(stale_keys)} stale key(s). Removing...')
    with h5py.File(out_path, 'a') as h5f:
        for key in tqdm(stale_keys, desc='Removing Stale Keys'):
            if key in h5f:
                del h5f[key]
    print(f'\n✅ Sync complete! Removed {len(stale_keys)} orphaned entries from dataset.h5.')

def append_new_songs(features_dir=FEATURES_DIR, out_path=HDF5_DATASET_PATH, cleanup_after_pack: bool=False):
    if not os.path.exists(out_path):
        print(f'❌ No database found at {out_path}. Run option 1 first.')
        return
    all_songs = sorted([f for f in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, f)) and any((file.endswith('.npy') for file in os.listdir(os.path.join(features_dir, f))))])
    needs_cover_update = []
    with h5py.File(out_path, 'r') as h5f:
        existing_keys = set(h5f.keys())
        for sid in existing_keys:
            song_dir = os.path.join(features_dir, sid)
            for cv in ('cover_2', 'cover_3', 'cover', 'original'):
                npy = os.path.join(song_dir, f'{cv}.npy')
                if os.path.exists(npy) and cv not in h5f[sid]:
                    needs_cover_update.append(sid)
                    break
    new_songs = [s for s in all_songs if s not in existing_keys]
    if not new_songs and (not needs_cover_update):
        print(f'✅ All {len(all_songs)} songs already up-to-date in database. Nothing to add.')
        return
    print(f'📊 Database has {len(existing_keys)} songs.')
    if new_songs:
        print(f'🆕 Found {len(new_songs)} new songs to add.')
    if needs_cover_update:
        print(f'🔄 Found {len(needs_cover_update)} existing songs needing missing cover updates.\n')
    added_covers = 0
    h5f = h5py.File(out_path, 'a')
    try:
        if new_songs:
            for (i, song_id) in enumerate(tqdm(new_songs, desc='Adding New Songs')):
                if i > 0 and i % 10 == 0:
                    h5f.close()
                    gc.collect()
                    h5f = h5py.File(out_path, 'a')
                song_dir = os.path.join(features_dir, song_id)
                group = h5f.require_group(song_id)
                song_ok = False
                for version in ['original', 'cover', 'cover_2', 'cover_3']:
                    npy_path = os.path.join(song_dir, f'{version}.npy')
                    if not os.path.exists(npy_path):
                        continue
                    try:
                        data = np.load(npy_path, allow_pickle=True).item()
                        cqt = data.get('harmonic_cqt')
                        if cqt is None:
                            cqt = data.get('cqt_features')
                        if cqt is None:
                            continue
                        cqt = cqt.astype(np.float16)
                        (n_segs, n_bins, n_frames) = cqt.shape
                        valid_mask = (cqt.mean(axis=(1, 2)) > 0.0001).astype(np.uint8)
                        chunk_shape = (1, n_bins, n_frames)
                        dset = group.create_dataset(version, data=cqt, compression='lzf', chunks=chunk_shape)
                        dset.attrs['shift_to_c'] = data.get('shift_to_c', 0)
                        dset.attrs['global_key'] = data.get('global_key', 'Unknown')
                        dset.attrs['valid_mask'] = valid_mask
                        song_ok = True
                        del data, cqt, valid_mask, chunk_shape
                        dset = None
                    except Exception as e:
                        print(f'\n  Error packing {song_id}/{version}: {e}')
                if not song_ok and song_id in h5f:
                    del h5f[song_id]
                h5f.flush()
                if song_ok and cleanup_after_pack:
                    shutil.rmtree(song_dir, ignore_errors=True)
        if needs_cover_update:
            for (i, song_id) in enumerate(tqdm(needs_cover_update, desc='Adding missing covers')):
                if i > 0 and i % 10 == 0:
                    h5f.close()
                    gc.collect()
                    h5f = h5py.File(out_path, 'a')
                song_dir = os.path.join(features_dir, song_id)
                group = h5f.require_group(song_id)
                for version in ['original', 'cover', 'cover_2', 'cover_3']:
                    if version in group:
                        continue
                    npy_path = os.path.join(song_dir, f'{version}.npy')
                    if not os.path.exists(npy_path):
                        continue
                    try:
                        data = np.load(npy_path, allow_pickle=True).item()
                        cqt = data.get('harmonic_cqt')
                        if cqt is None:
                            cqt = data.get('cqt_features')
                        if cqt is None:
                            continue
                        cqt = cqt.astype(np.float16)
                        (n_segs, n_bins, n_frames) = cqt.shape
                        valid_mask = (cqt.mean(axis=(1, 2)) > 0.0001).astype(np.uint8)
                        chunk_shape = (1, n_bins, n_frames)
                        dset = group.create_dataset(version, data=cqt, compression='lzf', chunks=chunk_shape)
                        dset.attrs['shift_to_c'] = data.get('shift_to_c', 0)
                        dset.attrs['global_key'] = data.get('global_key', 'Unknown')
                        dset.attrs['valid_mask'] = valid_mask
                        added_covers += 1
                        del data, cqt, valid_mask, chunk_shape
                        dset = None
                    except Exception as e:
                        print(f'\n  Error adding {song_id}/{version}: {e}')
                h5f.flush()
    finally:
        try:
            h5f.close()
        except:
            pass
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'\n✅ Done!')
    if new_songs:
        print(f'   Added {len(new_songs)} new songs.')
    if added_covers:
        print(f'   Added {added_covers} missing cover versions.')
    print(f'   Database now has {len(existing_keys) + len(new_songs)} total songs.')
    print(f'   Size: {size_mb:.1f} MB')

def _create_or_append(features_dir=FEATURES_DIR, out_path=HDF5_DATASET_PATH):
    if not os.path.exists(out_path):
        valid_songs = sorted([f for f in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, f)) and any((file.endswith('.npy') for file in os.listdir(os.path.join(features_dir, f))))])
        if not valid_songs:
            return 0
        print(f'[Watch] Creating dataset.h5 with {len(valid_songs)} ready songs...')
        written = 0
        h5f = h5py.File(out_path, 'w')
        try:
            for (i, song_id) in enumerate(tqdm(valid_songs, desc='Creating dataset.h5')):
                if i > 0 and i % 10 == 0:
                    h5f.close()
                    gc.collect()
                    h5f = h5py.File(out_path, 'a')
                song_dir = os.path.join(features_dir, song_id)
                group = h5f.require_group(song_id)
                song_ok = False
                for version in ['original', 'cover', 'cover_2', 'cover_3']:
                    npy_path = os.path.join(song_dir, f'{version}.npy')
                    if not os.path.exists(npy_path):
                        continue
                    try:
                        data = np.load(npy_path, allow_pickle=True).item()
                        cqt = data.get('harmonic_cqt')
                        if cqt is None:
                            cqt = data.get('cqt_features')
                        if cqt is None:
                            continue
                        cqt = cqt.astype(np.float16)
                        valid_mask = (cqt.mean(axis=(1, 2)) > 0.0001).astype(np.uint8)
                        dset = group.create_dataset(version, data=cqt, compression='lzf', chunks=(1, cqt.shape[1], cqt.shape[2]))
                        dset.attrs['shift_to_c'] = data.get('shift_to_c', 0)
                        dset.attrs['global_key'] = data.get('global_key', 'Unknown')
                        dset.attrs['valid_mask'] = valid_mask
                        song_ok = True
                        del data, cqt, valid_mask
                    except Exception as e:
                        print(f'\n  Error packing {song_id}/{version}: {e}')
                if not song_ok and song_id in h5f:
                    del h5f[song_id]
                else:
                    written += 1
                h5f.flush()
        finally:
            try:
                h5f.close()
            except:
                pass
        size_mb = os.path.getsize(out_path) / 1000000.0
        print(f'[Watch] Created dataset.h5 with {written} songs ({size_mb:.1f} MB)')
        return written
    else:
        all_songs = sorted([f for f in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, f)) and any((file.endswith('.npy') for file in os.listdir(os.path.join(features_dir, f))))])
        try:
            with h5py.File(out_path, 'r') as h5f:
                existing = set(h5f.keys())
                needs_cover_update = []
                for sid in h5f.keys():
                    song_dir = os.path.join(features_dir, sid)
                    for cv in ('cover_2', 'cover_3'):
                        npy = os.path.join(song_dir, f'{cv}.npy')
                        if os.path.exists(npy) and cv not in h5f[sid]:
                            needs_cover_update.append(sid)
                            break
        except Exception:
            existing = set()
            needs_cover_update = []
        new_songs = [s for s in all_songs if s not in existing]
        if not new_songs and (not needs_cover_update):
            return 0
        print(f'[Watch] Appending {len(new_songs)} new song(s) to dataset.h5...')
        written = 0
        _MAX_RETRIES = 8
        _RETRY_DELAY = 5
        for _attempt in range(_MAX_RETRIES):
            try:
                h5f = h5py.File(out_path, 'a')
                try:
                    for (i, song_id) in enumerate(tqdm(new_songs, desc='Appending to dataset.h5')):
                        if i > 0 and i % 10 == 0:
                            h5f.close()
                            gc.collect()
                            h5f = h5py.File(out_path, 'a')
                        song_dir = os.path.join(features_dir, song_id)
                        group = h5f.require_group(song_id)
                        song_ok = False
                        for version in ['original', 'cover', 'cover_2', 'cover_3']:
                            npy_path = os.path.join(song_dir, f'{version}.npy')
                            if not os.path.exists(npy_path):
                                continue
                            try:
                                data = np.load(npy_path, allow_pickle=True).item()
                                cqt = data.get('harmonic_cqt')
                                if cqt is None:
                                    cqt = data.get('cqt_features')
                                if cqt is None:
                                    continue
                                cqt = cqt.astype(np.float16)
                                valid_mask = (cqt.mean(axis=(1, 2)) > 0.0001).astype(np.uint8)
                                dset = group.create_dataset(version, data=cqt, compression='lzf', chunks=(1, cqt.shape[1], cqt.shape[2]))
                                dset.attrs['shift_to_c'] = data.get('shift_to_c', 0)
                                dset.attrs['global_key'] = data.get('global_key', 'Unknown')
                                dset.attrs['valid_mask'] = valid_mask
                                song_ok = True
                                del data, cqt, valid_mask
                            except Exception as e:
                                print(f'\n  Error packing {song_id}/{version}: {e}')
                        if not song_ok and song_id in h5f:
                            del h5f[song_id]
                        else:
                            written += 1
                        h5f.flush()
                    if needs_cover_update:
                        for (i, song_id) in enumerate(tqdm(needs_cover_update, desc='Adding new covers to dataset.h5')):
                            if i > 0 and i % 10 == 0:
                                h5f.close()
                                gc.collect()
                                h5f = h5py.File(out_path, 'a')
                            song_dir = os.path.join(features_dir, song_id)
                            group = h5f.require_group(song_id)
                            for version in ('cover_2', 'cover_3'):
                                if version in group:
                                    continue
                                npy_path = os.path.join(song_dir, f'{version}.npy')
                                if not os.path.exists(npy_path):
                                    continue
                                try:
                                    data = np.load(npy_path, allow_pickle=True).item()
                                    cqt = data.get('harmonic_cqt')
                                    if cqt is None:
                                        cqt = data.get('cqt_features')
                                    if cqt is None:
                                        continue
                                    cqt = cqt.astype(np.float16)
                                    valid_mask = (cqt.mean(axis=(1, 2)) > 0.0001).astype(np.uint8)
                                    dset = group.create_dataset(version, data=cqt, compression='lzf', chunks=(1, cqt.shape[1], cqt.shape[2]))
                                    dset.attrs['shift_to_c'] = data.get('shift_to_c', 0)
                                    dset.attrs['global_key'] = data.get('global_key', 'Unknown')
                                    dset.attrs['valid_mask'] = valid_mask
                                    written += 1
                                    del data, cqt, valid_mask
                                except Exception as e:
                                    print(f'\n  Error adding {song_id}/{version}: {e}')
                            h5f.flush()
                finally:
                    try:
                        h5f.close()
                    except:
                        pass
                gc.collect()
                break
            except OSError:
                if _attempt < _MAX_RETRIES - 1:
                    import time as _t
                    print(f'[Watch] dataset.h5 locked — retry {_attempt + 1}/{_MAX_RETRIES} in {_RETRY_DELAY}s')
                    _t.sleep(_RETRY_DELAY)
                else:
                    print(f'[Watch] dataset.h5 still busy after {_MAX_RETRIES} retries — skipping scan')
                    return 0
        import time as _t2
        _t2.sleep(0.5)
        total = 0
        try:
            with h5py.File(out_path, 'r') as h5f:
                total = len(h5f.keys())
        except Exception:
            pass
        size_mb = os.path.getsize(out_path) / 1000000.0
        print(f'[Watch] Appended {written} song(s). dataset.h5: {total} songs total ({size_mb:.1f} MB)')
        return written
if __name__ == '__main__':
    import argparse, time, signal as _signal
    import sys
    _STOP = False

    def _sigint(sig, frame):
        global _STOP
        _STOP = True
        print('\n\n🛑 Ctrl+C — stopping...')
        raise KeyboardInterrupt
    _signal.signal(_signal.SIGINT, _sigint)
    parser = argparse.ArgumentParser(description='HDF5 Database Manager — packs features/ NPY files into dataset.h5', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='Pipeline usage (run all 4 in separate terminals):\n  python script/download_dataset.py\n  python script/extract_features.py   --watch\n  python script/index_to_hdf5.py       --watch\n  python script/build_melody_index.py  --watch\n')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--watch', action='store_true', help='Watch mode: auto-append new songs every --interval seconds')
    mode.add_argument('--append', action='store_true', help='Non-interactive: append new songs once and exit')
    mode.add_argument('--create', action='store_true', help='Non-interactive: create or resume full database build')
    mode.add_argument('--sync', action='store_true', help='Remove stale entries (songs deleted from features/)')
    parser.add_argument('--interval', type=int, default=30, help='Seconds between scans in watch mode (default: 30)')
    parser.add_argument('--cleanup', action='store_true', help='Delete NPY folders from features/ after packing into HDF5 (saves disk space)')
    args = parser.parse_args()
    if args.watch:
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
        print('\n' + '=' * 70)
        print('[*] INDEX WATCH MODE -- auto-packing features -> dataset.h5')
        print(f'    Scanning every {args.interval}s  |  Press Ctrl+C to stop')
        print('=' * 70 + '\n')
        while not _STOP:
            try:
                if not os.path.exists(FEATURES_DIR) or not os.listdir(FEATURES_DIR):
                    sys.stdout.write('\r[Watch] Waiting for extracted features...   ')
                    sys.stdout.flush()
                else:
                    n = _create_or_append()
                    if n > 0:
                        print(f'\n[OK] Indexed {n} song(s). Scanning for more...\n')
                if not _STOP:
                    _db_total = 0
                    _db_mb = 0.0
                    try:
                        if os.path.exists(HDF5_DATASET_PATH):
                            _db_mb = os.path.getsize(HDF5_DATASET_PATH) / 1000000.0
                            with h5py.File(HDF5_DATASET_PATH, 'r') as _h:
                                _db_total = len(_h.keys())
                    except Exception:
                        pass
                    for i in range(args.interval):
                        if _STOP:
                            break
                        sys.stdout.write(f'\r[Watch] DB: {_db_total} songs ({_db_mb:.0f}MB) | Next scan in {args.interval - i}s   ')
                        sys.stdout.flush()
                        time.sleep(1)
                    sys.stdout.write('\n')
            except KeyboardInterrupt:
                break
        print('\n[OK] Watch mode stopped.')
    elif args.append:
        append_new_songs(cleanup_after_pack=args.cleanup)
    elif args.create:
        create_hdf5_database(cleanup_after_pack=args.cleanup)
    elif args.sync:
        sync_hdf5_database()
    else:
        print('HDF5 Database Manager')
        print('1. Create new database (or resume interrupted creation)')
        print('2. Sync existing database (remove stale entries after folder deletions)')
        print('3. Add new songs and missing covers (fastest — does not touch existing entries)')
        choice = input('Choose [1/2/3]: ').strip()
        _cleanup = input('Delete NPY folders after packing? (y/n, default n): ').strip().lower() == 'y'
        if _cleanup:
            print('⚠️  Cleanup ON — NPY folders will be deleted after packing.')
        if choice == '2':
            sync_hdf5_database()
        elif choice == '3':
            append_new_songs(cleanup_after_pack=_cleanup)
        else:
            create_hdf5_database(cleanup_after_pack=_cleanup)
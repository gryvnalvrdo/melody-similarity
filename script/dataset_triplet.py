import os 
import numpy as np 
import torch 
from torch .utils .data import Dataset ,DataLoader 
import random 
import gc 
from tqdm import tqdm 
import torch .nn .functional as F 
try :
    from config import FEATURES_DIR ,BATCH_SIZE ,HDF5_DATASET_PATH ,BINS_PER_OCTAVE 
except ImportError :
    from script .config import FEATURES_DIR ,BATCH_SIZE ,HDF5_DATASET_PATH ,BINS_PER_OCTAVE 
try :
    import h5py 
    H5PY_AVAILABLE =True 
except ImportError :
    H5PY_AVAILABLE =False 
class TripletDataset (Dataset ):
    def __init__ (self ,features_dir =FEATURES_DIR ,split ='train',train_ratio =0.8 ,
    cross_version_ratio =0.75 ):
        self .features_dir =features_dir 
        self .split =split 
        self .is_train =(split =='train')
        self .cross_version_ratio =cross_version_ratio 
        self .h5_path =HDF5_DATASET_PATH 
        self .use_h5 =H5PY_AVAILABLE and os .path .exists (self .h5_path )
        self .h5_handle =None 
        valid_songs :list [str ]=[]
        if self .use_h5 :
            with h5py .File (self .h5_path ,'r')as h5f :
                valid_songs =[str (k )for k in h5f .keys ()]
        else :
            for f in os .listdir (features_dir ):
                song_dir =os .path .join (features_dir ,f )
                if os .path .isdir (song_dir )and any (file .endswith ('.npy')for file in os .listdir (song_dir )):
                    valid_songs .append (f )
        all_songs :list [str ]=sorted (valid_songs )
        if not all_songs :
            self .song_ids =[]
            self .song_data ={}
            self .all_segments =[]
            self .cross_version_pairs ={}
            print (f"Warning: No valid features found in {features_dir}")
            return 
        split_idx =int (len (all_songs )*train_ratio )
        if split =='train':
            self .song_ids :list [str ]=all_songs [:split_idx ]
        else :
            self .song_ids =all_songs [split_idx :]
        self .song_data ={}
        self .all_segments :list [tuple [str ,str ,int ]]=[]
        self .cross_version_pairs ={}
        self .song_id_to_idx ={sid :idx for idx ,sid in enumerate (self .song_ids )}
        print (f"Loading {len(self.song_ids)} songs for {split} set (HDF5: {self.use_h5})...")
        if self .use_h5 :
            with h5py .File (self .h5_path ,'r')as h5f :
                for song_id in tqdm (self .song_ids ,desc =f"Loading HDF5 Metadata ({split})"):
                    grp =h5f [song_id ]
                    self .song_data [song_id ]={}
                    for version in ['original','cover','cover_2','cover_3']:
                        if version in grp :
                            dset =grp [version ]
                            n_segs =dset .shape [0 ]
                            self .song_data [song_id ][version ]={
                            'n_segments':n_segs ,
                            'shift_to_c':dset .attrs .get ('shift_to_c',0 ),
                            'global_key':dset .attrs .get ('global_key','Unknown')
                            }
                            if 'valid_mask'in dset .attrs :
                                valid_mask =dset .attrs ['valid_mask']
                                for seg_idx ,is_valid in enumerate (valid_mask ):
                                    if is_valid :
                                        self .all_segments .append ((str (song_id ),version ,int (seg_idx )))
                            else :
                                for seg_idx in range (n_segs ):
                                    chunk =dset [seg_idx ]
                                    if chunk .mean ()>1e-4 :
                                        self .all_segments .append ((str (song_id ),version ,int (seg_idx )))
        else :
            for i ,song_id in enumerate (tqdm (self .song_ids ,desc =f"Loading RAM ({split})")):
                if i >0 and i %50 ==0 :
                    gc .collect ()
                song_dir =os .path .join (features_dir ,song_id )
                self .song_data [song_id ]={}
                for version in ['original','cover','cover_2','cover_3']:
                    feature_path =os .path .join (song_dir ,f"{version}.npy")
                    if not os .path .exists (feature_path ):
                        continue 
                    try :
                        data =np .load (feature_path ,allow_pickle =True ).item ()
                        cqt =data .get ('harmonic_cqt')
                        if cqt is None :
                            cqt =data .get ('cqt_features')
                        if cqt is None :
                            continue 
                        self .song_data [song_id ][version ]={
                        'n_segments':len (cqt ),
                        'shift_to_c':data .get ('shift_to_c',0 ),
                        'global_key':data .get ('global_key','Unknown')
                        }
                        for seg_idx in range (len (cqt )):
                            if cqt [seg_idx ].mean ()>1e-4 :
                                self .all_segments .append ((str (song_id ),version ,int (seg_idx )))
                        del cqt 
                        del data 
                    except Exception as e :
                        print (f"Error loading {song_id}/{version}: {e}")
        cross_count =0 
        for song_id in self .song_ids :
            covers =[v for v in ['cover','cover_2','cover_3']
            if v in self .song_data .get (song_id ,{})]
            if 'original'in self .song_data .get (song_id ,{})and covers :
                self .cross_version_pairs [song_id ]=covers 
                cross_count +=1 
        avg_covers =(
        sum (len (v )for v in self .cross_version_pairs .values ())/max (cross_count ,1 )
        )
        print (f"Total segments: {len(self.all_segments)}")
        print (f"Cross-version pairs: {cross_count}/{len(self.song_ids)} songs "
        f"(avg {avg_covers:.1f} covers/song)")
    def __len__ (self ):
        return len (self .all_segments )
    def __del__ (self ):
        if self .h5_handle is not None :
            try :
                self .h5_handle .close ()
            except Exception :
                pass 
            self .h5_handle =None 
    def shift_cqt (self ,cqt ,shift_semitones ):
        _bins_per_semitone =BINS_PER_OCTAVE //12 
        shift_bins =shift_semitones *_bins_per_semitone 
        if shift_bins ==0 :
            return cqt 
        return torch .roll (cqt ,shift_bins ,dims =0 )
    def apply_time_stretch (self ,cqt ,factor ):
        if abs (factor -1.0 )<0.01 :
            return cqt 
        n_bins ,n_frames =cqt .shape 
        new_width =int (n_frames *factor )
        cqt_tensor =cqt .unsqueeze (0 )
        stretched_tensor =F .interpolate (
        cqt_tensor ,size =new_width ,mode ='linear',align_corners =False 
        )
        stretched =stretched_tensor .squeeze (0 )
        if new_width >=n_frames :
            start =(new_width -n_frames )//2 
            return stretched [:,start :start +n_frames ]
        else :
            pad_total =n_frames -new_width 
            pad_l =pad_total //2 
            pad_r =pad_total -pad_l 
            return F .pad (stretched ,(pad_l ,pad_r ),mode ='constant',value =0.0 )
    def augment (self ,cqt ,shift_semitones ,stretch_factor ):
        cqt =self .shift_cqt (cqt ,shift_semitones )
        cqt =self .apply_time_stretch (cqt ,stretch_factor )

        wiggle =random .randint (-2 ,2 )
        if wiggle !=0 :
            n_bins ,n_frames =cqt .shape 
            if wiggle >0 :

                pad =torch .zeros (n_bins ,wiggle ,dtype =cqt .dtype )
                cqt =torch .cat ([pad ,cqt [:,:-wiggle ]],dim =1 )
            else :

                w =abs (wiggle )
                pad =torch .zeros (n_bins ,w ,dtype =cqt .dtype )
                cqt =torch .cat ([cqt [:,w :],pad ],dim =1 )
        if random .random ()<0.5 :
            cutoff =random .randint (6 ,20 )
            cqt [-cutoff :,:]=0.0 
        noise =torch .randn_like (cqt )*random .uniform (0.01 ,0.05 )
        cqt =torch .clamp (cqt +noise ,0.0 ,1.0 )
        if random .random ()<0.6 :
            n_bins ,n_frames =cqt .shape 
            num_freq_masks =random .randint (1 ,2 )
            for _ in range (num_freq_masks ):
                f_mask_width =random .randint (1 ,12 )
                f0 =random .randint (0 ,max (0 ,n_bins -f_mask_width ))
                cqt [f0 :f0 +f_mask_width ,:]=0.0 
            num_time_masks =random .randint (1 ,2 )
            for _ in range (num_time_masks ):
                t_mask_width =random .randint (5 ,20 )
                t0 =random .randint (0 ,max (0 ,n_frames -t_mask_width ))
                cqt [:,t0 :t0 +t_mask_width ]=0.0 
        return cqt 
    def timbre_augment (self ,cqt :torch .Tensor )->torch .Tensor :
        n_bins ,_ =cqt .shape 
        _bpo =BINS_PER_OCTAVE 
        if random .random ()<0.50 :
            tilt =random .uniform (-0.55 ,0.55 )
            weights =torch .linspace (1.0 -tilt *0.6 ,1.0 +tilt *0.6 ,n_bins )
            if cqt .is_cuda :
                weights =weights .cuda ()
            cqt =torch .clamp (cqt *weights .unsqueeze (1 ),0.0 ,1.0 )
        if random .random ()<0.35 :

            octave =random .randint (1 ,min (6 ,n_bins //_bpo -1 ))
            start ,end =octave *_bpo ,min (octave *_bpo +_bpo ,n_bins )
            boost =random .uniform (1.15 ,1.60 )
            cqt [start :end ,:]=torch .clamp (cqt [start :end ,:]*boost ,0.0 ,1.0 )
        if random .random ()<0.30 :

            suppress =random .uniform (0.30 ,0.75 )
            cqt [:_bpo ,:]=torch .clamp (cqt [:_bpo ,:]*suppress ,0.0 ,1.0 )
        return cqt 
    def _get_cqt (self ,song_id ,version ,seg_idx ):
        if self .use_h5 :
            if self .h5_handle is None :
                self .h5_handle =h5py .File (self .h5_path ,'r')
            return self .h5_handle [song_id ][version ][seg_idx ].astype (np .float32 )
        else :
            feature_path =os .path .join (self .features_dir ,song_id ,f"{version}.npy")
            data =np .load (feature_path ,allow_pickle =True ).item ()
            cqt =data .get ('harmonic_cqt')
            if cqt is None :
                cqt =data .get ('cqt_features')
            return cqt [seg_idx ].astype (np .float32 )
    def _find_best_match_cqt (self ,anchor_cqt :np .ndarray ,song_id :str ,
    version :str ,lo :int ,hi :int ):
        try :
            if self .use_h5 :
                if self .h5_handle is None :
                    self .h5_handle =h5py .File (self .h5_path ,'r')
                candidates =self .h5_handle [song_id ][version ][lo :hi +1 ].astype (np .float32 )
            else :
                feature_path =os .path .join (self .features_dir ,song_id ,f"{version}.npy")
                data =np .load (feature_path ,allow_pickle =True ).item ()
                cqt =data .get ('harmonic_cqt')or data .get ('cqt_features')
                if cqt is None :
                    return lo ,None ,0.0 
                candidates =cqt [lo :hi +1 ].astype (np .float32 )
            if len (candidates )==0 :
                return lo ,None ,0.0 
            anchor_flat =anchor_cqt .flatten ()
            anchor_norm =np .linalg .norm (anchor_flat )
            if anchor_norm <1e-8 :
                rand_i =random .randint (0 ,len (candidates )-1 )
                return lo +rand_i ,candidates [rand_i ],0.0 
            anchor_flat =anchor_flat /anchor_norm 
            cands_flat =candidates .reshape (len (candidates ),-1 )
            norms =np .linalg .norm (cands_flat ,axis =1 ,keepdims =True )
            norms =np .where (norms <1e-8 ,1.0 ,norms )
            sims =(cands_flat /norms )@anchor_flat 
            best_local =int (np .argmax (sims ))
            return lo +best_local ,candidates [best_local ],float (sims [best_local ])
        except MemoryError :
            rand_i =random .randint (lo ,hi )
            return rand_i ,None ,0.0 
        except Exception as _e :
            print (f"\n  [!] _find_best_match_cqt({song_id}/{version}): {_e} -- using random fallback")
            rand_i =random .randint (lo ,hi )
            return rand_i ,None ,0.0 
    def __getitem__ (self ,idx ):
        song_id ,version ,seg_idx =self .all_segments [idx ]
        base_cqt =self ._get_cqt (song_id ,version ,seg_idx )
        base_cqt_t =torch .from_numpy (base_cqt )
        if self .is_train :
            shared_shift =random .randint (-5 ,5 )                                
            shared_stretch =random .uniform (0.75 ,1.30 )
            anchor =self .augment (base_cqt_t ,shared_shift ,shared_stretch )
        else :
            anchor =base_cqt_t .clone ()
        has_cross =song_id in self .cross_version_pairs 
        use_cross =has_cross and (not self .is_train or random .random ()<self .cross_version_ratio )
        if use_cross :
            all_versions =list (self .song_data [song_id ].keys ())
            candidate_versions =[v for v in all_versions if v !=version ]
            if not candidate_versions :
                candidate_versions =['original']if 'original'in self .song_data [song_id ]else all_versions 
            other_ver =random .choice (candidate_versions )
            other_data =self .song_data [song_id ][other_ver ]
            other_n =other_data ['n_segments']
            rel_pos =seg_idx /max (self .song_data [song_id ][version ]['n_segments']-1 ,1 )
            center =int (rel_pos *(other_n -1 ))

            window =max (2 ,int (other_n *0.30 ))
            lo ,hi =max (0 ,center -window ),min (other_n -1 ,center +window )
            pos_idx ,pos_cqt ,best_sim =self ._find_best_match_cqt (
            base_cqt ,song_id ,other_ver ,lo ,hi 
            )

            _MIN_CROSS_SIM =0.10 
            if pos_cqt is None or best_sim <_MIN_CROSS_SIM :
                if self .is_train :
                    positive =self .augment (base_cqt_t ,shared_shift ,shared_stretch )
                else :
                    positive =base_cqt_t .clone ()
            else :
                pos_cqt_t =torch .from_numpy (pos_cqt )
                if self .is_train :
                    pos_stretch =random .uniform (0.80 ,1.25 )
                    if random .random ()<0.50 :
                        cover_shift =random .randint (-5 ,5 )                                
                    else :
                        cover_shift =shared_shift 
                    pos_cqt_t =self .timbre_augment (pos_cqt_t )
                    positive =self .augment (pos_cqt_t ,cover_shift ,pos_stretch )
                else :
                    positive =pos_cqt_t 
        else :
            if self .is_train :
                positive =self .augment (base_cqt_t ,shared_shift ,shared_stretch )
            else :
                positive =base_cqt_t .clone ()
        song_idx =torch .tensor (self .song_id_to_idx .get (song_id ,-1 ),dtype =torch .long )
        return anchor ,positive ,song_idx 
def create_data_loaders (features_dir =FEATURES_DIR ,batch_size =BATCH_SIZE ):
    train_dataset =TripletDataset (features_dir ,split ='train')
    val_dataset =TripletDataset (features_dir ,split ='val')
    use_h5 =H5PY_AVAILABLE and os .path .exists (HDF5_DATASET_PATH )
    if os .name =='nt':
        n_workers =4 if use_h5 else 0 
    else :
        n_workers =8 if use_h5 else 0 
    extra_kwargs :dict ={}
    if n_workers >0 :
        extra_kwargs ['prefetch_factor']=4 
        extra_kwargs ['persistent_workers']=True 
    samples_per_epoch =min (50000 ,len (train_dataset ))
    train_sampler =torch .utils .data .RandomSampler (
    train_dataset ,replacement =True ,num_samples =samples_per_epoch 
    )
    train_loader =DataLoader (
    train_dataset ,
    batch_size =batch_size ,
    sampler =train_sampler ,
    num_workers =n_workers ,
    pin_memory =True ,
    drop_last =True ,
    **extra_kwargs 
    )
    val_samples =min (25600 ,len (val_dataset ))
    val_sampler =torch .utils .data .RandomSampler (
    val_dataset ,replacement =False ,num_samples =val_samples 
    )
    val_loader =DataLoader (
    val_dataset ,
    batch_size =batch_size ,
    sampler =val_sampler ,
    num_workers =n_workers ,
    pin_memory =True ,
    **extra_kwargs 
    )
    return train_loader ,val_loader 
if __name__ =="__main__":
    print ("\n== TESTING DATASET ==\n")
    dataset =TripletDataset (FEATURES_DIR ,split ='train')
    print (f"Songs: {len(dataset.song_ids)}")
    print (f"Segments: {len(dataset)}")
    print (f"Cross-version pairs: {len(dataset.cross_version_pairs)}")
    if len (dataset )>0 :
        anchor ,positive ,song_idx =dataset [0 ]
        print (f"Anchor: {anchor.shape}")
        print (f"Positive: {positive.shape}")
        print (f"Song index: {song_idx.item()}")
        print (f"Value range: [{anchor.min():.3f}, {anchor.max():.3f}]")
        print ("\nTest passed!")

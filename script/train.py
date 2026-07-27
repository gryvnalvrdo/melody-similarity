import copy 
import os 
import torch 
import torch .nn as nn 
import torch .nn .functional as F 
import torch .optim as optim 
from tqdm import tqdm 
import numpy as np 
import argparse 
import shutil 
try :
    from config import (
    MODELS_DIR ,DEVICE ,BATCH_SIZE ,LEARNING_RATE ,WEIGHT_DECAY ,
    EPOCHS ,PATIENCE ,MIN_DELTA ,TEMPERATURE 
    )
    from model import MelodySimilarityModel 
    from dataset_triplet import create_data_loaders 
except ImportError :
    from script .config import (
    MODELS_DIR ,DEVICE ,BATCH_SIZE ,LEARNING_RATE ,WEIGHT_DECAY ,
    EPOCHS ,PATIENCE ,MIN_DELTA ,TEMPERATURE 
    )
    from script .model import MelodySimilarityModel 
    from script .dataset_triplet import create_data_loaders 
ACCUMULATION_STEPS =16
WARMUP_EPOCHS =5
MOMENTUM =0.995
QUEUE_SIZE =8192

LABEL_MODE ='song'

HNM_POOL_SIZE    = 512
HNM_TOPK_RATIO   = 0.3
HNM_START_EPOCH  = 5
HNM_UPDATE_FREQ  = 2

class HardNegativeSampler:
    def __init__(self, pool_size: int = HNM_POOL_SIZE, topk_ratio: float = HNM_TOPK_RATIO):
        self.pool_size  = pool_size
        self.topk_ratio = topk_ratio
        self.bank_embs  = None
        self.bank_sids  = None
        self.ready      = False

    @torch.no_grad()
    def update_bank(self, model, loader, device, max_batches: int = 50):
        model.eval()
        all_embs, all_sids = [], []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            a, _, sid = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            emb = F.normalize(model(a).float(), dim=1)
            all_embs.append(emb.cpu())
            all_sids.append(sid.cpu())
        model.train()
        if all_embs:
            self.bank_embs = torch.cat(all_embs, dim=0)
            self.bank_sids = torch.cat(all_sids, dim=0)
            self.ready     = True

    @torch.no_grad()
    def get_hard_negatives(self, anchor_embs: torch.Tensor,
                           anchor_sids: torch.Tensor,
                           device) -> torch.Tensor:
        if not self.ready or self.bank_embs is None:
            return None

        bank    = F.normalize(self.bank_embs.to(device), dim=1)
        anchors = F.normalize(anchor_embs.detach(), dim=1)
        sids    = anchor_sids.to(device)

        sim_mat  = torch.mm(anchors, bank.T)
        bank_sids = self.bank_sids.to(device)

        hard_negs = []
        topk_n    = max(1, int(self.pool_size * self.topk_ratio))

        for i in range(sim_mat.size(0)):
            same_song = (bank_sids == sids[i])
            sim_row   = sim_mat[i].clone()
            sim_row[same_song] = -2.0

            n_cand = min(topk_n, (~same_song).sum().item())
            if n_cand == 0:
                hard_negs.append(bank[0])
                continue
            _, topk_idx = torch.topk(sim_row, k=n_cand)
            chosen_idx  = topk_idx[torch.randint(0, n_cand, (1,))].item()
            hard_negs.append(bank[chosen_idx])

        return torch.stack(hard_negs, dim=0)

class SupConLoss (nn .Module ):
    def __init__ (self ,temperature :float =0.07 ):
        super ().__init__ ()
        self .temperature =temperature 
        self .last_metrics ={}
    def forward (self ,anchor ,positive ,song_indices =None ,
    queue =None ,queue_sids =None ):
        device =anchor .device 
        N =anchor .size (0 )
        anchor =F .normalize (anchor ,p =2 ,dim =1 )
        positive =F .normalize (positive ,p =2 ,dim =1 )
        features =torch .cat ([anchor ,positive ],dim =0 )
        if song_indices is not None :
            labels =torch .cat ([song_indices ,song_indices ])
        else :
            labels =torch .cat ([
            torch .arange (N ,device =device ),
            torch .arange (N ,device =device ),
            ])
        if queue is not None and queue .shape [0 ]>0 :
            contrast =torch .cat ([features ,queue ],dim =0 )
            c_labels =torch .cat ([labels ,
            queue_sids if queue_sids is not None 
            else torch .full ((queue .shape [0 ],),-1 ,device =device )])
        else :
            contrast =features 
            c_labels =labels 
        M =contrast .shape [0 ]
        sim =torch .matmul (features ,contrast .T )/self .temperature 
        mask_pos =(labels .unsqueeze (1 )==c_labels .unsqueeze (0 )).float ()
        self_mask =torch .zeros (2 *N ,M ,device =device )
        self_mask [:,:2 *N ].fill_diagonal_ (1.0 )
        mask_pos =mask_pos *(1.0 -self_mask )
        if queue is not None and queue .shape [0 ]>0 :
            Q =queue .shape [0 ]
            fn_queue_mask =(labels .unsqueeze (1 )==queue_sids .unsqueeze (0 )if queue_sids is not None else torch .zeros (2 *N ,Q ,dtype =torch .bool ,device =device ))
            denom_mask_inbatch =self_mask .bool ()
            denom_mask_queue =torch .zeros (2 *N ,M ,dtype =torch .bool ,device =device )
            denom_mask_queue [:,2 *N :]=fn_queue_mask 
            denom_mask =denom_mask_inbatch |denom_mask_queue 
        else :
            denom_mask =self_mask .bool ()
        sim_max ,_ =torch .max (sim ,dim =1 ,keepdim =True )
        sim =sim -sim_max .detach ()
        exp_sim =torch .exp (sim ).masked_fill (denom_mask ,0.0 )
        log_denom =torch .log (exp_sim .sum (dim =1 ,keepdim =True )+1e-9 )
        log_probs =sim -log_denom 
        num_pos =mask_pos .sum (dim =1 )
        valid =num_pos >0 
        per_loss =-(mask_pos *log_probs ).sum (dim =1 )
        per_loss =torch .where (
        valid ,per_loss /(num_pos +1e-9 ),torch .zeros_like (per_loss )
        )
        loss =per_loss [valid ].mean ()if valid .any ()else torch .tensor (0.0 ,device =device )
        with torch .no_grad ():
            pos_sim =(anchor *positive ).sum (dim =-1 )
            neg_mat =torch .matmul (features ,features .T )
            eye2 =torch .eye (2 *N ,dtype =torch .bool ,device =device )
            pp_mask =torch .zeros (2 *N ,2 *N ,dtype =torch .bool ,device =device )
            pp_mask [torch .arange (N ),torch .arange (N ,2 *N )]=True 
            pp_mask [torch .arange (N ,2 *N ),torch .arange (N )]=True 
            neg_mat .masked_fill_ (eye2 |pp_mask ,0.0 )
            n_neg_inbatch =2 *N *(2 *N -2 )
            neg_sim_inbatch =neg_mat .sum ()/max (n_neg_inbatch ,1 )
            if queue is not None and queue .shape [0 ]>0 :
                q_sim =torch .matmul (features ,queue .T )
                if queue_sids is not None :
                    fn_q =(labels .unsqueeze (1 )==queue_sids .unsqueeze (0 ))
                    q_sim .masked_fill_ (fn_q ,0.0 )
                    n_neg_q =max ((~fn_q ).sum ().item (),1 )
                else :
                    n_neg_q =queue .shape [0 ]*2 *N 
                neg_sim_queue =q_sim .sum ()/n_neg_q 
                n_total =n_neg_inbatch +n_neg_q 
                neg_sim =(neg_sim_inbatch *n_neg_inbatch +neg_sim_queue *n_neg_q )/n_total 
            else :
                neg_sim =neg_sim_inbatch 
            self .last_metrics ={
            'pos_sim':float (pos_sim .mean ()),
            'neg_sim':float (neg_sim ),
            'gap':float (pos_sim .mean ()-neg_sim ),
            'loss_c':float (loss ),
            }
        return loss 
    def get_metrics (self ):
        return self .last_metrics 
class MomentumEncoder :
    def __init__ (self ,online_model :nn .Module ,momentum :float =0.995 ,device ='cpu'):
        self .momentum =momentum 
        self .encoder =copy .deepcopy (online_model ).to (device )
        for p in self .encoder .parameters ():
            p .requires_grad_ (False )
        self .encoder .eval ()
    @torch .no_grad ()
    def update (self ,online_model :nn .Module ):
        for p_k ,p_q in zip (self .encoder .parameters (),
        online_model .parameters ()):
            p_k .data .mul_ (self .momentum ).add_ (p_q .data ,
            alpha =1.0 -self .momentum )
    @torch .no_grad ()
    def encode (self ,x :torch .Tensor )->torch .Tensor :
        self .encoder .eval ()
        return self .encoder (x )
class NegativeQueue :
    def __init__ (self ,queue_size :int =4096 ,dim :int =256 ,device ='cpu'):
        self .queue_size =queue_size 
        self .queue =F .normalize (torch .randn (queue_size ,dim ),dim =1 ).to (device )
        self .queue_sids =torch .full ((queue_size ,),-1 ,
        dtype =torch .long ,device =device )
        self .ptr =0 
        self .filled =0 
    def enqueue (self ,embeddings :torch .Tensor ,song_ids :torch .Tensor ):
        n =embeddings .shape [0 ]
        idx =torch .arange (self .ptr ,self .ptr +n ,
        device =self .queue .device )%self .queue_size 
        self .queue [idx ]=embeddings .detach ().to (self .queue .device )
        self .queue_sids [idx ]=song_ids .detach ().to (self .queue .device )
        self .ptr =(self .ptr +n )%self .queue_size 
        self .filled =min (self .filled +n ,self .queue_size )
    def get (self ):
        return self .queue [:self .filled ],self .queue_sids [:self .filled ]
def compute_embedding_stats (model ,loader ,device ,max_batches =20 ):
    model .eval ()
    all_emb =[]
    with torch .no_grad ():
        for i ,batch in enumerate (loader ):
            if i >=max_batches :
                break 
            emb =model (batch [0 ].to (device ))
            all_emb .append (emb .cpu ())
    if not all_emb :
        return {}
    all_emb =torch .cat (all_emb ,dim =0 )
    mean_var =all_emb .var (dim =0 ).mean ().item ()
    n =min (all_emb .size (0 ),200 )
    sample =all_emb [torch .randperm (all_emb .size (0 ))[:n ]]
    sim =F .cosine_similarity (sample .unsqueeze (1 ),sample .unsqueeze (0 ),dim =-1 )
    off_diag =~torch .eye (n ,dtype =torch .bool )
    return {
    'embedding_variance':mean_var ,
    'mean_pairwise_similarity':sim [off_diag ].mean ().item (),
    }
def train_epoch (model ,train_loader ,criterion ,optimizer ,device ,
scaler =None ,max_batches =None ,
momentum_enc :MomentumEncoder =None ,
neg_queue :NegativeQueue =None ):
    model .train ()
    total_loss =0 
    num_batches =0 
    optimizer .zero_grad ()
    pbar =tqdm (train_loader ,desc ="Training",leave =False )
    mb_anchors :list [torch .Tensor ]=[]
    mb_positives :list [torch .Tensor ]=[]
    mb_sids :list [torch .Tensor ]=[]
    for i ,batch in enumerate (pbar ):
        if max_batches and i >=max_batches :
            break 
        anchor ,positive ,sid =batch [0 ],batch [1 ],batch [2 ]
        mb_anchors .append (anchor .to (device ,non_blocking =True ))
        mb_positives .append (positive .to (device ,non_blocking =True ))
        mb_sids .append (sid .to (device ,non_blocking =True ))
        if (i +1 )%ACCUMULATION_STEPS !=0 and (i +1 )!=len (train_loader ):
            continue 
        a_embs ,p_embs =[],[]
        _dev =next (model .parameters ()).device .type 
        with torch .no_grad ():
            for a ,p in zip (mb_anchors ,mb_positives ):
                with torch .amp .autocast (_dev ,enabled =scaler is not None ):
                    a_embs .append (model (a ).float ())
                    p_embs .append (model (p ).float ())
        large_a =torch .cat (a_embs ,dim =0 ).requires_grad_ (True )
        large_p =torch .cat (p_embs ,dim =0 ).requires_grad_ (True )
        large_sid =torch .cat (mb_sids ,dim =0 )
        _q_emb ,_q_sid =neg_queue .get ()if neg_queue is not None else (None ,None )

        if LABEL_MODE =='pair':
            _loss_sid ,_loss_qsid =None ,None 
        elif LABEL_MODE =='hybrid':
            if num_batches %2 ==0 :
                _loss_sid ,_loss_qsid =None ,None 
            else :
                _loss_sid ,_loss_qsid =large_sid ,_q_sid 
        else :
            _loss_sid ,_loss_qsid =large_sid ,_q_sid 
        with torch .amp .autocast (_dev ,enabled =scaler is not None ):
            loss =criterion (large_a ,large_p ,_loss_sid ,
            queue =_q_emb ,queue_sids =_loss_qsid )
        loss .backward ()
        sizes =[a .size (0 )for a in mb_anchors ]
        a_grads =large_a .grad .split (sizes )
        p_grads =large_p .grad .split (sizes )
        for a ,p ,ag ,pg in zip (mb_anchors ,mb_positives ,a_grads ,p_grads ):

            with torch .amp .autocast (_dev ,enabled =scaler is not None ):
                a_emb =model (a ).float ()
            surrogate_a =(a_emb *ag .detach ()).sum ()
            if scaler is not None :
                scaler .scale (surrogate_a ).backward ()
            else :
                surrogate_a .backward ()

            with torch .amp .autocast (_dev ,enabled =scaler is not None ):
                p_emb =model (p ).float ()
            surrogate_p =(p_emb *pg .detach ()).sum ()
            if scaler is not None :
                scaler .scale (surrogate_p ).backward ()
            else :
                surrogate_p .backward ()
        if scaler is not None :
            scaler .unscale_ (optimizer )
            torch .nn .utils .clip_grad_norm_ (model .parameters (),max_norm =1.0 )
            scaler .step (optimizer )
            scaler .update ()
        else :
            torch .nn .utils .clip_grad_norm_ (model .parameters (),max_norm =1.0 )
            optimizer .step ()
        optimizer .zero_grad ()
        if momentum_enc is not None :
            momentum_enc .update (model )
            if neg_queue is not None :
                with torch .no_grad ():
                    mom_embs =torch .cat ([
                    F .normalize (momentum_enc .encode (p ),dim =-1 )
                    for p in mb_positives 
                    ],dim =0 )

                neg_queue .enqueue (mom_embs ,large_sid )
        total_loss +=loss .item ()
        num_batches +=1 
        mb_anchors ,mb_positives ,mb_sids =[],[],[]
        if hasattr (criterion ,'get_metrics'):
            m =criterion .get_metrics ()
            pbar .set_postfix ({
            'loss':f'{loss.item():.4f}',
            'gap':f'{m.get("gap", 0):.3f}',
            'p_sim':f'{m.get("pos_sim", 0):.3f}',
            'n_sim':f'{m.get("neg_sim", 0):.3f}',
            })
    return total_loss /max (num_batches ,1 )
def validate (model ,val_loader ,criterion ,device ,use_amp =False ,
queue =None ,queue_sids =None ):
    model .eval ()
    total_loss =0 
    num_batches =0 
    pos_sims ,neg_sims =[],[]
    acc_a ,acc_p ,acc_sid =[],[],[]
    with torch .no_grad ():
        for i ,batch in enumerate (val_loader ):
            a ,p ,sid =batch [0 ].to (device ),batch [1 ].to (device ),batch [2 ].to (device )
            with torch .amp .autocast (device .type if hasattr (device ,'type')else str (device ),enabled =use_amp ):
                acc_a .append (model (a ))
                acc_p .append (model (p ))
            acc_sid .append (sid )
            if (i +1 )%ACCUMULATION_STEPS ==0 or (i +1 )==len (val_loader ):
                la =torch .cat (acc_a )
                lp =torch .cat (acc_p )
                ls =torch .cat (acc_sid )

                if LABEL_MODE =='pair':
                    _val_sid ,_val_qsid =None ,None 
                elif LABEL_MODE =='hybrid':

                    if num_batches %2 ==0 :
                        _val_sid ,_val_qsid =None ,None 
                    else :
                        _val_sid ,_val_qsid =ls ,queue_sids 
                else :
                    _val_sid ,_val_qsid =ls ,queue_sids 
                with torch .amp .autocast (device .type if hasattr (device ,'type')else str (device ),enabled =use_amp ):
                    loss =criterion (la ,lp ,_val_sid ,
                    queue =queue ,queue_sids =_val_qsid )
                total_loss +=loss .item ()
                num_batches +=1 
                m =criterion .get_metrics ()
                pos_sims .append (m .get ('pos_sim',0 ))
                neg_sims .append (m .get ('neg_sim',0 ))
                acc_a ,acc_p ,acc_sid =[],[],[]
    return (total_loss /max (num_batches ,1 ),
    float (np .mean (pos_sims ))if pos_sims else 0.0 ,
    float (np .mean (neg_sims ))if neg_sims else 0.0 )
def train (
epochs =EPOCHS ,
batch_size =BATCH_SIZE ,
lr =LEARNING_RATE ,
patience =PATIENCE ,
resume_from =None ,
max_batches =None ,
momentum =MOMENTUM ,
queue_size =QUEUE_SIZE ,
**kwargs 
):
    eff_batch =batch_size *ACCUMULATION_STEPS 
    print ("="*70 )
    print ("MELODY SIMILARITY TRAINING  --  MoCo HNM + SupCon + FN Masking")
    print ("="*70 )
    print (f"Device        : {DEVICE}")
    print (f"Epochs        : {epochs}  |  Patience: {patience}")
    print (f"Batch size    : {batch_size}  (effective: {eff_batch})")
    print (f"LR            : {lr:.1e}  (warmup {WARMUP_EPOCHS} epochs -> cosine)")
    print (f"EMA momentum  : {momentum}  (MoCo-style momentum encoder)")
    print (f"Queue size    : {queue_size}  (hard negatives from momentum encoder)")
    print (f"In-batch neg  : {eff_batch * 2}  +  queue: {queue_size}  "
    f"= {eff_batch * 2 + queue_size} total negatives (max)")
    _label_desc ={
    'pair':'NT-Xent (anchor↔positive only) — enables cross-song similarity',
    'song':'SupCon  (song-level labels)    — strict cover detection',
    'hybrid':'Hybrid  (alternating per step) — balanced approach',
    }
    print (f"Label mode    : {LABEL_MODE} -- {_label_desc.get(LABEL_MODE, '')}")
    print ("="*70 )
    torch .backends .cudnn .benchmark =True 
    print ("\nCreating model...")
    model =MelodySimilarityModel ().to (DEVICE )
    print (f"[Model] {sum(p.numel() for p in model.parameters()):,} params")
    print ("\nLoading data...")
    train_loader ,val_loader =create_data_loaders (batch_size =batch_size )
    print (f"Train batches : {len(train_loader)}")
    print (f"Val batches   : {len(val_loader)}")
    optimizer =optim .AdamW (model .parameters (),lr =lr ,weight_decay =WEIGHT_DECAY )
    cosine_scheduler =optim .lr_scheduler .CosineAnnealingLR (
    optimizer ,T_max =(epochs -WARMUP_EPOCHS ),eta_min =1e-6 )
    criterion =SupConLoss (temperature =TEMPERATURE ).to (DEVICE )
    _amp_dtype =DEVICE .type 
    scaler =torch .amp .GradScaler (_amp_dtype )if DEVICE .type =='cuda'else None 
    use_amp =(DEVICE .type =='cuda')
    if use_amp :
        print ("AMP (mixed precision) : enabled")
    momentum_enc =MomentumEncoder (model ,momentum =momentum ,device =DEVICE )
    neg_queue =NegativeQueue (queue_size =queue_size ,dim =256 ,device =DEVICE )
    print (f"[HNM] Momentum encoder  : EMA tau={momentum}")
    print (f"[HNM] Negative queue    : {queue_size} x 256D  "
    f"({queue_size * 256 * 4 / 1024 / 1024:.1f} MB)")
    use_hnm = kwargs.get('use_hnm', True)
    hnm_sampler = HardNegativeSampler() if use_hnm else None
    if use_hnm:
        print(f"[HNM] Hard Negative Sampler : AKTIF (start_epoch={HNM_START_EPOCH}, "
              f"pool={HNM_POOL_SIZE}, topk_ratio={HNM_TOPK_RATIO})")
    else:
        print(f"[HNM] Hard Negative Sampler : NONAKTIF")
    start_epoch =0 
    best_loss =float ('inf')
    best_gap =float ('-inf')
    if resume_from and os .path .exists (resume_from ):
        print (f"\nResuming from {resume_from}")
        ckpt =torch .load (resume_from ,map_location =DEVICE ,weights_only =False )
        model .load_state_dict (ckpt ['model_state_dict'])
        optimizer .load_state_dict (ckpt ['optimizer_state_dict'])
        start_epoch =ckpt ['epoch']+1 
        best_loss =ckpt .get ('best_loss',float ('inf'))
        best_gap =ckpt .get ('best_gap',float ('-inf'))
        if 'cosine_scheduler_state_dict'in ckpt and start_epoch >=WARMUP_EPOCHS :
            cosine_scheduler .load_state_dict (ckpt ['cosine_scheduler_state_dict'])
        for p_k ,p_q in zip (momentum_enc .encoder .parameters (),model .parameters ()):
            p_k .data .copy_ (p_q .data )
        print (f"Resumed epoch {start_epoch}, best gap {best_gap:.4f}")
    print ("\nInitial embedding stats...")
    s =compute_embedding_stats (model ,train_loader ,DEVICE )
    print (f"   Variance    : {s.get('embedding_variance', 0):.6f}")
    print (f"   Pairwise sim: {s.get('mean_pairwise_similarity', 0):.4f}")
    print ("\n"+"="*70 )
    print ("STARTING TRAINING")
    print ("="*70 )
    os .makedirs (MODELS_DIR ,exist_ok =True )
    no_improve =0 
    collapse_warn =0 
    emb_var =0.0 
    pair_sim =0.0 
    history ={'train_loss':[],'val_loss':[],'pos_sim':[],'neg_sim':[],'gap':[]}
    try :
        for epoch in range (start_epoch ,epochs ):
            if epoch <WARMUP_EPOCHS :
                wlr =lr *(0.1 +0.9 *epoch /WARMUP_EPOCHS )
                for pg in optimizer .param_groups :
                    pg ['lr']=wlr 
                print (f"\nEpoch {epoch+1}/{epochs}  "
                f"(warmup LR: {wlr:.1e}, "
                f"queue: {neg_queue.filled}/{queue_size})")
            else :
                print (f"\nEpoch {epoch+1}/{epochs}  "
                f"(LR: {optimizer.param_groups[0]['lr']:.1e}, "
                f"queue: {neg_queue.filled}/{queue_size})")
            if hnm_sampler is not None and epoch >= HNM_START_EPOCH:
                if (epoch - HNM_START_EPOCH) % HNM_UPDATE_FREQ == 0:
                    hnm_sampler.update_bank(model, train_loader, DEVICE,
                                            max_batches=HNM_POOL_SIZE // batch_size + 1)
                    print(f"   [HNM] Bank updated: {len(hnm_sampler.bank_embs) if hnm_sampler.bank_embs is not None else 0} embeddings")

            train_loss =train_epoch (
            model ,train_loader ,criterion ,optimizer ,DEVICE ,
            scaler =scaler ,max_batches =max_batches ,
            momentum_enc =momentum_enc ,neg_queue =neg_queue )
            if epoch >=WARMUP_EPOCHS :
                cosine_scheduler .step ()
            _q_emb ,_q_sid =neg_queue .get ()if neg_queue is not None else (None ,None )
            val_loss ,pos_sim ,neg_sim =validate (
            model ,val_loader ,criterion ,DEVICE ,use_amp =use_amp ,
            queue =_q_emb ,queue_sids =_q_sid )
            gap =float (pos_sim -neg_sim )
            print (f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print (f"   Pos Sim: {pos_sim:.4f} | Neg Sim: {neg_sim:.4f} | Gap: {gap:.4f}")
            if (epoch -start_epoch )%5 ==0 :
                stats =compute_embedding_stats (model ,train_loader ,DEVICE ,max_batches =10 )
                emb_var =stats .get ('embedding_variance',0 )
                pair_sim =stats .get ('mean_pairwise_similarity',0 )
                print (f"   Emb Var: {emb_var:.6f} | Pairwise Sim: {pair_sim:.4f}")
            else :
                pass 
            if emb_var <1e-5 or pair_sim >0.95 :
                collapse_warn +=1 
                print (f"   [!] WARNING: possible collapse #{collapse_warn}")
            else :
                collapse_warn =max (0 ,collapse_warn -1 )
            for k ,v in zip (['train_loss','val_loss','pos_sim','neg_sim','gap'],
            [train_loss ,val_loss ,pos_sim ,neg_sim ,gap ]):
                history [k ].append (v )
            if val_loss <best_loss :
                best_loss =val_loss 
            if gap >best_gap +MIN_DELTA :
                best_gap =gap 
                no_improve =0 
                ckpt_path =os .path .join (MODELS_DIR ,"best_model.pt")
                torch .save ({
                'epoch':epoch ,
                'model_state_dict':model .state_dict (),
                'optimizer_state_dict':optimizer .state_dict (),
                'cosine_scheduler_state_dict':cosine_scheduler .state_dict (),
                'best_loss':best_loss ,
                'best_gap':best_gap ,
                'gap':gap ,
                'loss_phase':'SupCon+MoCo-HNM',
                },ckpt_path )
                print (f"   SAVED best_model.pt  (gap: {gap:.4f}, val_loss: {val_loss:.4f})")
            else :
                no_improve +=1 
                print (f"   No improvement {no_improve}/{patience}  (best gap: {best_gap:.4f})")
            if no_improve >=patience :
                print (f"\nEarly stopping at epoch {epoch+1}")
                break 
            if (epoch +1 )%10 ==0 :
                torch .save ({
                'epoch':epoch ,
                'model_state_dict':model .state_dict (),
                'optimizer_state_dict':optimizer .state_dict (),
                'cosine_scheduler_state_dict':cosine_scheduler .state_dict (),
                'best_loss':best_loss ,
                'best_gap':best_gap ,
                },os .path .join (MODELS_DIR ,f"checkpoint_epoch_{epoch+1}.pt"))
    except KeyboardInterrupt :
        print ("\n\n[STOP] Interrupted -- saving emergency checkpoint...")
        ep_safe =epoch if 'epoch'in locals ()else start_epoch 
        torch .save ({
        'epoch':ep_safe ,
        'model_state_dict':model .state_dict (),
        'optimizer_state_dict':optimizer .state_dict (),
        'cosine_scheduler_state_dict':cosine_scheduler .state_dict (),
        'best_loss':best_loss ,'best_gap':best_gap ,
        'history':history ,'interrupted':True ,
        },os .path .join (MODELS_DIR ,"interrupted_checkpoint.pt"))
        print ("[OK] Emergency checkpoint saved.")
    ep_safe =epoch if 'epoch'in locals ()else start_epoch 
    torch .save ({
    'epoch':ep_safe ,
    'model_state_dict':model .state_dict (),
    'optimizer_state_dict':optimizer .state_dict (),
    'cosine_scheduler_state_dict':cosine_scheduler .state_dict (),
    'best_loss':best_loss ,'best_gap':best_gap ,'history':history ,
    },os .path .join (MODELS_DIR ,"final_model.pt"))
    best_path =os .path .join (MODELS_DIR ,"best_model.pt")
    if not os .path .exists (best_path ):
        shutil .copy2 (os .path .join (MODELS_DIR ,"final_model.pt"),best_path )
    print ("\n"+"="*70 )
    print ("TRAINING COMPLETE")
    print ("="*70 )
    print (f"Best val loss : {best_loss:.4f}")
    if history ['gap']:
        print (f"Best gap      : {max(history['gap']):.4f}")
    print (f"Models saved  : {MODELS_DIR}")
    return model ,history 
if __name__ =="__main__":
    parser =argparse .ArgumentParser (
    description ="Melody similarity — NT-Xent + MoCo HNM + False Negative Masking")
    parser .add_argument ("--epochs",type =int ,default =EPOCHS )
    parser .add_argument ("--batch-size",type =int ,default =BATCH_SIZE )
    parser .add_argument ("--lr",type =float ,default =LEARNING_RATE )
    parser .add_argument ("--resume",type =str ,default =None )
    parser .add_argument ("--momentum",type =float ,default =MOMENTUM ,
    help ="EMA momentum for momentum encoder (default: 0.995)")
    parser .add_argument ("--queue-size",type =int ,default =QUEUE_SIZE ,
    help ="Hard negative queue size (default: 4096)")
    parser .add_argument ("--test",action ="store_true",
    help ="5-epoch smoke-test with small queue")
    args =parser .parse_args ()
    if args .test :
        train (epochs =5 ,batch_size =8 ,queue_size =256 )
    else :
        train (
        epochs =args .epochs ,
        batch_size =args .batch_size ,
        lr =args .lr ,
        resume_from =args .resume ,
        momentum =args .momentum ,
        queue_size =args .queue_size ,
        )

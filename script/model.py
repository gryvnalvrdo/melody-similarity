import torch 
import torch .nn as nn 
import torch .nn .functional as F 
try :
    from config import (
    N_CQT_BINS ,EMBEDDING_DIM ,CNN_CHANNELS ,
    LSTM_HIDDEN ,LSTM_LAYERS ,DROPOUT ,DEVICE ,BOTTLENECK_DIM 
    )
except ImportError :
    from script .config import (
    N_CQT_BINS ,EMBEDDING_DIM ,CNN_CHANNELS ,
    LSTM_HIDDEN ,LSTM_LAYERS ,DROPOUT ,DEVICE ,BOTTLENECK_DIM 
    )

_FREQ_POOL_BANDS =max (21 ,N_CQT_BINS //4 )
class ResBlock (nn .Module ):
    def __init__ (self ,in_channels ,out_channels ,stride =1 ):
        super ().__init__ ()
        self .conv1 =nn .Conv2d (in_channels ,out_channels ,3 ,stride =stride ,padding =1 ,bias =False )
        self .bn1 =nn .InstanceNorm2d (out_channels ,affine =True )
        self .conv2 =nn .Conv2d (out_channels ,out_channels ,3 ,stride =1 ,padding =1 ,bias =False )
        self .bn2 =nn .InstanceNorm2d (out_channels ,affine =True )
        self .skip =nn .Sequential ()
        if stride !=1 or in_channels !=out_channels :
            self .skip =nn .Sequential (
            nn .Conv2d (in_channels ,out_channels ,1 ,stride =stride ,bias =False ),
            nn .InstanceNorm2d (out_channels ,affine =True )
            )
    def forward (self ,x ):
        identity =self .skip (x )
        out =F .leaky_relu (self .bn1 (self .conv1 (x )),0.1 )
        out =self .bn2 (self .conv2 (out ))
        out =F .leaky_relu (out +identity ,0.1 )
        return out 
class AttentionPooling (nn .Module ):
    def __init__ (self ,in_features ):
        super ().__init__ ()
        self .attention =nn .Sequential (
        nn .Linear (in_features ,in_features //2 ),
        nn .Tanh (),
        nn .Linear (in_features //2 ,1 )
        )
    def forward (self ,x ):
        attn_weights =self .attention (x )
        attn_weights =F .softmax (attn_weights ,dim =1 )
        pooled =(x *attn_weights ).sum (dim =1 )
        return pooled 
class MelodySimilarityModel (nn .Module ):
    def __init__ (
    self ,
    n_input_bins =N_CQT_BINS ,
    embedding_dim =EMBEDDING_DIM ,
    cnn_channels =CNN_CHANNELS ,
    lstm_hidden =LSTM_HIDDEN ,
    lstm_layers =LSTM_LAYERS ,
    dropout =DROPOUT 
    ):
        super ().__init__ ()
        self .n_input_bins =n_input_bins 
        self .embedding_dim =embedding_dim 
        self .input_conv =nn .Sequential (
        nn .Conv2d (1 ,cnn_channels [0 ],kernel_size =(3 ,7 ),stride =(1 ,2 ),padding =(1 ,3 ),bias =False ),
        nn .InstanceNorm2d (cnn_channels [0 ],affine =True ),
        nn .LeakyReLU (0.1 )
        )
        self .res_blocks =nn .ModuleList ()
        in_ch =cnn_channels [0 ]
        for i ,out_ch in enumerate (cnn_channels ):
            stride =(1 ,1 )if i ==0 else (1 ,2 )
            self .res_blocks .append (ResBlock (in_ch ,out_ch ,stride =stride ))
            in_ch =out_ch 
        self .freq_pool =nn .AdaptiveAvgPool2d ((_FREQ_POOL_BANDS ,1 ))
        freq_dim =_FREQ_POOL_BANDS 
        raw_lstm_input_size =cnn_channels [-1 ]*freq_dim 
        self .bottleneck_dim =BOTTLENECK_DIM 
        self .bottleneck =nn .Sequential (
        nn .Linear (raw_lstm_input_size ,self .bottleneck_dim ),
        nn .LayerNorm (self .bottleneck_dim ),
        nn .LeakyReLU (0.1 )
        )
        self .lstm =nn .LSTM (
        input_size =self .bottleneck_dim ,
        hidden_size =lstm_hidden ,
        num_layers =lstm_layers ,
        batch_first =True ,
        bidirectional =True ,
        dropout =dropout if lstm_layers >1 else 0 
        )
        self .lstm_norm =nn .LayerNorm (lstm_hidden *2 )
        fc_input_dim =lstm_hidden *2 
        self .attention_pool =AttentionPooling (fc_input_dim )
        self .proj =nn .Sequential (
        nn .Linear (fc_input_dim ,fc_input_dim *2 ,bias =False ),
        nn .LayerNorm (fc_input_dim *2 ),
        nn .ReLU (inplace =True ),
        nn .Linear (fc_input_dim *2 ,embedding_dim ,bias =False )
        )
        for m in self .modules ():
            if isinstance (m ,nn .Conv2d ):
                nn .init .kaiming_normal_ (m .weight ,mode ='fan_out',nonlinearity ='leaky_relu')
            elif isinstance (m ,nn .InstanceNorm2d ):
                if hasattr (m ,'weight')and getattr (m ,'weight')is not None :
                    nn .init .constant_ (m .weight ,1 )
                if hasattr (m ,'bias')and getattr (m ,'bias')is not None :
                    nn .init .constant_ (m .bias ,0 )
            elif isinstance (m ,nn .Linear ):
                nn .init .orthogonal_ (m .weight )
            elif isinstance (m ,nn .LSTM ):
                for name ,param in m .named_parameters ():
                    if 'weight_ih'in name :
                        nn .init .orthogonal_ (param .data )
                    elif 'weight_hh'in name :
                        nn .init .orthogonal_ (param .data )
                    elif 'bias'in name :
                        param .data .fill_ (0 )
                        n =param .size (0 )
                        param .data [n //4 :n //2 ].fill_ (1.0 )
        total_params =sum (p .numel ()for p in self .parameters ())
        print (f"[Model] Lightweight CRNN Bottleneck: {total_params:,} params")
        print (f"[Model] CNN: {cnn_channels} -> Bottleneck {self.bottleneck_dim} -> LSTM {lstm_hidden}x{lstm_layers} -> {embedding_dim}D")
    def forward (self ,x ):
        if x .dim ()==3 :
            x =x .unsqueeze (1 )
        x =self .input_conv (x )
        for block in self .res_blocks :
            x =block (x )
        x =x .permute (0 ,3 ,1 ,2 )
        batch ,time ,channels ,freq =x .shape 
        x =x .reshape (batch *time ,channels ,freq ,1 )
        x =self .freq_pool (x )
        x =x .reshape (batch ,time ,channels *_FREQ_POOL_BANDS )
        x =self .bottleneck (x )
        lstm_out ,_ =self .lstm (x )
        lstm_out =self .lstm_norm (lstm_out )
        global_feat =self .attention_pool (lstm_out )
        embedding =self .proj (global_feat )
        return F .normalize (embedding ,p =2 ,dim =1 )
if __name__ =="__main__":
    batch_size =4 
    n_frames =646 
    x =torch .randn (batch_size ,N_CQT_BINS ,n_frames ).to (DEVICE )
    model =MelodySimilarityModel ().to (DEVICE )
    model .eval ()
    with torch .no_grad ():
        emb =model (x )
    print (f"Output: {emb.shape}")
    print (f"Norm: {torch.norm(emb, dim=1).mean():.4f}")

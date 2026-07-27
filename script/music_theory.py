import numpy as np 
CHROMATIC_SCALE =['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
ENHARMONIC_MAP ={
'Db':'C#','D#':'Eb','Gb':'F#','G#':'Ab','A#':'Bb',
'Cb':'B','B#':'C','E#':'F','Fb':'E',
'C':'C','C#':'C#','D':'D','Eb':'Eb','E':'E',
'F':'F','F#':'F#','G':'G','Ab':'Ab','A':'A',
'Bb':'Bb','B':'B'
}
MAJOR_PROFILE =[5.0 ,2.0 ,3.5 ,2.0 ,4.5 ,4.0 ,2.0 ,4.5 ,2.0 ,3.5 ,1.5 ,4.0 ]
MINOR_PROFILE =[5.0 ,2.0 ,3.5 ,4.5 ,2.0 ,4.0 ,2.0 ,4.5 ,3.5 ,2.0 ,1.5 ,4.0 ]
MAJOR_INTERVALS =[2 ,2 ,1 ,2 ,2 ,2 ,1 ]
MINOR_INTERVALS =[2 ,1 ,2 ,2 ,1 ,2 ,2 ]
def get_chromatic_index (note ):
    normalized =ENHARMONIC_MAP .get (note ,None )
    if normalized is not None and normalized in CHROMATIC_SCALE :
        return CHROMATIC_SCALE .index (normalized )
    if note in CHROMATIC_SCALE :
        return CHROMATIC_SCALE .index (note )
    return -1 
def generate_scale (root ,intervals ):
    start_idx =get_chromatic_index (root )
    if start_idx ==-1 :
        return []
    scale_indices =[start_idx ]
    current =start_idx 
    for interval in intervals [:-1 ]:
        current =(current +interval )%12 
        scale_indices .append (current )
    return [CHROMATIC_SCALE [i ]for i in scale_indices ]
SCALES ={}
for _root in CHROMATIC_SCALE :
    SCALES [f"{_root} Major"]=generate_scale (_root ,MAJOR_INTERVALS )
    SCALES [f"{_root} Minor"]=generate_scale (_root ,MINOR_INTERVALS )
_ENHARMONIC_KEYS ={
'Db Major':'C# Major','Db Minor':'C# Minor',
'D# Major':'Eb Major','D# Minor':'Eb Minor',
'Gb Major':'F# Major','Gb Minor':'F# Minor',
'G# Major':'Ab Major','G# Minor':'Ab Minor',
'A# Major':'Bb Major','A# Minor':'Bb Minor',
}
for _alias ,_canonical in _ENHARMONIC_KEYS .items ():
    if _canonical in SCALES :
        SCALES [_alias ]=SCALES [_canonical ]
def get_scale_notes (key_name ):
    if ' Min'in key_name and 'Minor'not in key_name :
        key_name =key_name .replace (' Min',' Minor')
    if ' Maj'in key_name and 'Major'not in key_name :
        key_name =key_name .replace (' Maj',' Major')
    if key_name in SCALES :
        return SCALES [key_name ]
    parts =key_name .split (' ')
    if len (parts )==2 :
        root ,mode =parts 
        idx =get_chromatic_index (root )
        if idx !=-1 :
            canonical_root =CHROMATIC_SCALE [idx ]
            canonical_key =f"{canonical_root} {mode}"
            if canonical_key in SCALES :
                return SCALES [canonical_key ]
    return []
def detect_key_with_profile (chroma_sum ):
    chroma =np .array (chroma_sum ).flatten ()
    if chroma .shape [0 ]!=12 :
        return "Unknown",0.0 
    chroma_norm =(chroma -np .mean (chroma ))/(np .std (chroma )+1e-9 )
    scores =[]
    major_prof =np .array (MAJOR_PROFILE )
    major_prof_norm =(major_prof -np .mean (major_prof ))/(np .std (major_prof )+1e-9 )
    for i in range (12 ):
        target_profile =np .roll (major_prof_norm ,i )
        score =np .dot (chroma_norm ,target_profile )
        scores .append ((score ,f"{CHROMATIC_SCALE[i]} Major"))
    minor_prof =np .array (MINOR_PROFILE )
    minor_prof_norm =(minor_prof -np .mean (minor_prof ))/(np .std (minor_prof )+1e-9 )
    for i in range (12 ):
        target_profile =np .roll (minor_prof_norm ,i )
        score =np .dot (chroma_norm ,target_profile )
        scores .append ((score ,f"{CHROMATIC_SCALE[i]} Minor"))
    scores .sort (key =lambda x :x [0 ],reverse =True )
    best_score ,best_key =scores [0 ]
    runner_up_score =scores [1 ][0 ]
    margin_confidence =best_score -runner_up_score 
    return best_key ,float (margin_confidence )
def get_key_from_cqt (cqt_mag ,sr =None ):
    try :
        from config import BINS_PER_OCTAVE as _BPO 
    except ImportError :
        try :
            from script .config import BINS_PER_OCTAVE as _BPO 
        except ImportError :
            _BPO =12 
    cqt_sum =np .sum (cqt_mag ,axis =1 )
    n_bins =cqt_sum .shape [0 ]
    chroma =np .zeros (12 )
    for i in range (n_bins ):

        semitone =i *12 //_BPO 
        octave =i //_BPO 
        weight =2.0 if 3 <=octave <=5 else 1.0 
        chroma [semitone %12 ]+=cqt_sum [i ]*weight 
    return detect_key_with_profile (chroma )

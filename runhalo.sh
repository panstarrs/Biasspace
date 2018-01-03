#dir=/home/hpan/data/MYWAVES/Galform_Out/v2.7.0/aquarius_trees/medi-SURFS-VELOCIraptor/GALFORM.SURFS.BestModel.SMBHheateff.0.03
wdir=/Users/hengxingpan/Work/anaconda2/bin
idir=/Users/hengxingpan/Work/danail/data/medi/
odir=./output/B$5/H_$4

$wdir/python ./vhxyz.py -outdir $odir -indir $idir -zlist $5 -lbox 210 -h0 0.6751 -bands "$1" "$2" "$3" "$4"
#$wdir/python ./dhxyz.py -outdir $odir -indir $idir -zlist $5 -lbox 210 -h0 0.6751 -bands "$1" "$2" "$3" "$4"

#$wdir/python ./bias_tot.py -outdir $odir/B$5_ -indir $idir -zlist $5 -bands "$1" "$2" "$3" "$4"

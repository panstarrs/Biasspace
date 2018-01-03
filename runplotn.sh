#dir=/home/hpan/data/MYWAVES/Galform_Out/v2.7.0/aquarius_trees/medi-SURFS-VELOCIraptor/GALFORM.SURFS.BestModel.SMBHheateff.0.03
wdir=/Users/hengxingpan/Work/anaconda2/bin
idir=./output/B$5/G_$4/$6
odir=$idir

$wdir/python ./bias_normalx.py -outdir $odir -indir $idir -zlist $5 -bands "$1" "$2" "$3" "$4"

#dir=/home/hpan/data/MYWAVES/Galform_Out/v2.7.0/aquarius_trees/medi-SURFS-VELOCIraptor/GALFORM.SURFS.BestModel.SMBHheateff.0.03
wdir=/Users/hengxingpan/Work/anaconda2/bin
idir=/Users/hengxingpan/Work/danail/data/medi/

odir=./output/B$5/G_$4/$6/

$wdir/python ./gclf.py -outdir $odir/f_h/ -indir $idir -zlist $5 -lbox 210 -vlist '34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33' -h0 0.6751 -obs F -bands "$1" "$2" "$3" "$4" "$6"
$wdir/python ./gxyz.py -outdir $odir/p_c/ -indir $idir -zlist $5 -lbox 210 -vlist '34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33' -h0 0.6751 -bands "$1" "$2" "$3" "$4" "$6"

#$wdir/python ./bias_tot.py -outdir $odir/B$5_ -indir $idir -zlist $5 -bands "$1" "$2" "$3" "$4"

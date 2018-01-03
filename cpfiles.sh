wdir=/home/hpan/ProCorr/Bias/output
outdir=/Users/hengxingpan/Work/danail/ProCorr/Bias/
#wdir=$outdir

#for i in 113; do {
#for i in 113; do {
for i in 199 156 131 113; do {

  for j in 0.5; do {
  #for j in 0.2 0.5 1; do {
    #for k in centra; do
    #for k in sat orp; do
    for k in sat_cen; do
    #for k in centra sat_cen; do
    #for k in orp_sat_cen; do
      #echo $wdir/B$i/G_$j/$k/f_h
      scp pleiades:$wdir/B$i/G_$j/$k/p_c/*.txt $outdir/output/B$i/G_$j/$k/p_c/
      #scp pleiades:$wdir/B$i/G_$j/$k/f_h/*.txt $outdir/output/B$i/G_$j/$k/f_h/

      #mkdir -p $outdir/output/B$i/G_$j/$k/f_h
      #mkdir -p $outdir/output/B$i/G_$j/$k/p_c

      #rm $outdir/output/B$i/G_$j/$k/p_c/*
      #rm -r $outdir/output/B$i/G_$j/$k
    done

    #rm $outdir/output/B$i/G_$j/orp_sat_cen/p_c/L*_1*_p.txt
    #rm $outdir/output/B$i/G_$j/orp_sat_cen/p_c/L*_9*_p.txt
    #scp pleiades:$wdir/B$i/G_$j/orp_sat_cen/p_c/*_p.txt $outdir/output/B$i/G_$j/orp_sat_cen/p_c/

    #for l in p_c; do
    #scp pleiades:$wdir/B$i/G_$j/orp_sat_cen/$l/*tb.txt $outdir/output/B$i/G_$j/orp_sat_cen/$l/
    #done

    #rm  $outdir/output/B$i/G_$j/orp_sat_cen/p_c/*
    #rm  $outdir/output/B$i/G_$j/orp_sat_cen/f_h/*

    #scp pleiades:$wdir/B$i/G_$j/sat_cen/p_c/*.txt $outdir/output/B$i/G_$j/sat_cen/p_c/
    #scp pleiades:$wdir/B$i/G_$j/orp_sat/p_c/*.txt $outdir/output/B$i/G_$j/orp_sat/p_c/
    #scp pleiades:$wdir/B$i/G_$j/orp_sat/f_h/*.txt $outdir/output/B$i/G_$j/orp_sat/f_h/


    #scp pleiades:$wdir/B$i/G_$j/centra/p_c/*.txt $outdir/output/B$i/G_$j/centra/p_c/
    #scp pleiades:$wdir/B$i/G_$j/centra/f_h/*.txt $outdir/output/B$i/G_$j/centra/f_h/

    #scp pleiades:$wdir/B$i/G_$j/orp_sat_cen/p_c/*.txt $outdir/output/B$i/G_$j/orp_sat_cen/p_c/

    #scp pleiades:$wdir/B$i/G_$j/centra/f_h/* $outdir/output/B$i/G_$j/centra/f_h/
    #scp pleiades:$wdir/B$i/G_$j/orp_sat_cen/f_h/*.txt $outdir/output/B$i/G_$j/orp_sat_cen/f_h/
    #scp pleiades:$wdir/B$i/G_$j/orp_sat_cen/p_c/*.txt $outdir/output/B$i/G_$j/orp_sat_cen/p_c/

    #rm -r $outdir/output/B$i/G_$j/centra
    #scp -r pleiades:$wdir/B$i/G_$j/centra $outdir/output/B$i/G_$j/

    #mkdir -p $wdir/B$i/H_$j

    #mkdir -p $outdir/test/linp/B$i/H_$j
    #scp pleiades:$wdir/B$i/H_$j/*f.txt $outdir/test/linp/B$i/H_$j
    #scp pleiades:$wdir/B$i/H_$j/*p.txt $outdir/test/linp/B$i/H_$j

    #scp pleiades:$wdir/B$i/H_$j/*.txt $outdir/output/B$i/H_$j/

  } done
  #scp pleiades:$wdir/B$i/P_256/L210_N512_$i'_'p.txt $outdir/output/B$i/P_256/

} done

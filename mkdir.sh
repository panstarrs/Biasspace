wdir=/Users/hengxingpan/Work/danail/ProCorr/Bias/output
#wdir=/home/hpan/ProCorr/Bias/output

#for i in 113; do {
for i in 199 156 131 113; do {

  for j in 0.2; do {
  #for j in 0.2 0.5 1; do {
    for k in centra; do
    #for k in sat_cen; do
    #for k in sat orp; do
    #for k in orp_sat; do
    #for k in centra sat_cen orp_sat_cen; do
      #echo $wdir/B$i/G_$j/$k/f_h
      #mkdir -p $wdir/B$i/G_$j/$k/f_h
      #mkdir -p $wdir/B$i/G_$j/$k/p_c

      #rm $wdir/B$i/G_$j/$k/f_h/*
      rm $wdir/B$i/G_$j/$k/p_c/*10.*
    done

    #scp -r orp_wdir/B$i/G_$j/orp_sat_cen
    #mkdir -p $wdir/B$i/H_1
    #rm $wdir/B$i/H_$j/*

  } done
  #mkdir -p $wdir/B$i/T
  #mv $wdir/B$i/*.txt $wdir/B$i/T
  #mv $wdir/B$i/*.png $wdir/B$i/T
  #mv $wdir/B$i/particle_256 $wdir/B$i/P_256
  #mv $wdir/B$i/particle_512 $wdir/B$i/P_512

} done

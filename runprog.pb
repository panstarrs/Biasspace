#!/bin/sh

#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=1:compute
##PBS -l nodes=1:ppn=1:visual
##PBS -l mem=128gb

#PBS -N galf

date
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
wdir=/Users/hengxingpan/Work/danail/ProCorr
dir=$wdir/Bias/output/B$3/G_$2/$4/p_c

echo $dir

if [ $(echo "$2 == 1" | bc) -ne 0 ] ; then
#if [ "$2" -eq "1" ] ; then

  for filename in $dir/$1*0 ; do
      echo $filename
      $wdir/procorr $filename -input m2 -ncells 256 -output pdxl -interpolation 2
      #./procorr $dir/$file -input 1 -output x -sidelength 210.0
  done

else

  for filename in $dir/$1*5 ; do
      echo $filename
      $wdir/procorr $filename -input m2 -ncells 256 -output pdxl -interpolation 2
      #./procorr $dir/$file -input 1 -output x -sidelength 210.0
  done

fi


#declare -a  prelist=("mhhalo" "mstars" "L_tot" "mag_SDSS-r")

#if [ $3 -eq 1 ]
#then
#  #declare -a  prelist=("vhalo")
#  declare -a  prelist=("mhhalo")
#
#else
#  declare -a  prelist=("mhhalo" "mstars" "L_tot" "mag_SDSS-g" "mag_SDSS-i")
#fi
#
#index=0 # 0,1,2,3,4
#for index in $1 ; do {
#  for prefix in "${!prelist[@]}"; do {
#    #if [ $prefix -eq $index ]
#    if [ $prefix -ne $index ]
#    then
#      continue
#    fi
#  
#    for filename in $dir/${prelist[$prefix]}*0 ; do
#      echo $filename
#      $wdir/procorr $filename -input m2 -ncells 256 -output pdxl -interpolation 2
#      #./procorr $dir/$file -input 1 -output x -sidelength 210.0
#    done
#  
#  } done
#
#} done
#wait
date

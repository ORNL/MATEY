#!/bin/bash

jobscripts=submit_batch_4.sh
declare -a arr=("submit_batch_cades_32.sh"  "submit_batch_cades_64.sh"  "submit_batch_cades.sh"  "submit_batch_cades_t2.sh"  "submit_batch_cades_t3.sh")

for jobscripts in "${arr[@]}";
do
   for i in $(seq 1 1);
   do
	if [ $i -eq 1 ]; then
                echo sbatch $jobscripts
    		RES=$(sbatch $jobscripts)
   	 else
    		echo sbatch --dependency=afterany:${RES##* } $jobscripts
		#RES=$(sbatch --dependency=afterany:${RES##* } $jobscripts)
    	fi
    	echo $RES
   done
done


#!/bin/bash

jobscripts=submit.sh

for i in $(seq 1 1);
do
    if [ $i -eq 1 ]; then
    	RES=$(sbatch $jobscripts)
    	#RES=$(sbatch -dependency=afterany:1631092 $jobscripts)
    else
    	echo sbatch --dependency=afterany:${RES##* } $jobscripts
	RES=$(sbatch --dependency=afterany:${RES##* } $jobscripts)
    fi
    echo $RES
done


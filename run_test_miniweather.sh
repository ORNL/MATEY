#!/bin/bash
config_file=mpp_avit_ti_config_miniweather.yaml
config_new=mpp_avit_ti_config.yaml
job_script=submit_batch.sh
optm=adam
lr=1e-3
lr=1e-4

declare -a tokentests=("8" "16" "32" )   
declare -a tokeninputs=("[[8, 8]]" "[[16, 16]]" "[[32, 32]]")   

nrefines=4
nrefines=8
nrefines=16

#declare -a tokentests=("adap-t2-$nrefines")
#declare -a tokeninputs=("[[8, 8], [32, 32]]")

#declare -a tokentests=("adap-hybrid-$nrefines")
#declare -a tokeninputs=("[[8, 8], [32, 32]]")

declare -a tokentests=("adap-t2-$nrefines-sts")
declare -a tokeninputs=("[[8, 8], [32, 32]]")

declare -a tokentests=("adap-samp-t2-$nrefines-sts")
declare -a tokeninputs=("[[8, 8], [32, 32]]")

testlength=${#tokentests[@]}

JOBDIR=runs_${optm}_miniweather
mkdir $PWD/$JOBDIR
echo $JOBDIR
cd $JOBDIR

# use for loop to read all values and indexes
for (( i=0; i<${testlength}; i++ ));
do
  echo "index: $i, token: ${tokentests[$i]}, ${tokeninputs[$i]}"
  tokendir=tokens_${tokentests[$i]}
  mkdir $tokendir
  cd $tokendir
  #config_new=$config_file
  cp ../../config/$config_file $config_new
  sed -i -e "s/exp_dir: '~\/MPP_red_token'/exp_dir: .\//g" $config_new
  sed -i -e "s/optimizer: 'adan'/optimizer: '$optm'/g" $config_new
  sed -i -e "s/learning_rate: -1/learning_rate: $lr/g" $config_new
  sed -i -e "s/scheduler: 'cosine'/scheduler: 'warmuponly'/g" $config_new
  #sed -i -e "s/warmup_steps: 1000/warmup_steps: 500/g" $config_new
  sed -i -e "s/patch_size: \[\[16, 16\]\]/patch_size: ${tokeninputs[$i]}/g" $config_new
  sed -i -e "s/batch_size: 32/batch_size: 64/g" $config_new
  sed -i -e "s/epoch_size: 200/epoch_size: 96/g" $config_new
  sed -i -e "s/num_data_workers: 6/num_data_workers: 2/g" $config_new
  #sed -i -e "s/enable_amp: !!bool False/enable_amp: !!bool True/g" $config_new
  if [[ $tokendir == *"adap"* ]]; then
    sed -i -e "s/adaptive: !!bool False/adaptive: !!bool True/g" $config_new
    sed -i -e "s/nrefines: 0 /nrefines: $nrefines /g" $config_new
  fi
  if [[ $tokendir == *"sts"* ]]; then
    sed -i -e "s/sts_model: !!bool False/sts_model: !!bool True/g" $config_new
  fi
  if [[ $tokendir == *"samp"* ]]; then
    sed -i -e "s/adap_samp: !!bool False/adap_samp: !!bool True/g" $config_new
  fi
  sed -i -e "s/max_epochs: 800/max_epochs: 2000/g" $config_new


  job_new=$job_script
  cp ../../$job_script $job_new
  sed -i -e "s/#SBATCH -J demo/#SBATCH -J ${tokentests[$i]}/g" $job_new
  sed -i -e "s/yaml_config=.\/config\/mpp_avit_ti_config.yaml/yaml_config=$config_new/g" $job_new
  sed -i -e "s/train_basic.py/..\/..\/train_basic.py/g" $job_new

  #sbatch $job_new
  # submit multiple sequential jobs
  cp ../../submit_sequential_jobs.sh submit_sequential_jobs.sh
  bash submit_sequential_jobs.sh
  cd ..

done


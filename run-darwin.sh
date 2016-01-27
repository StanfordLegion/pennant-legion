export GASNET_USE_XRC=0
export OMPI_MCA_mpi_paffinity_alone=0 
export OMPI_MCA_rmaps_base_loadbalance=1
export OMPI_MCA_btl="sm,openib,self"
export OMPI_MCA_mpi_warn_on_fork=0
export LD_LIBRARY_PATH=${HOME}/local/lib 
export GASNET_BACKTRACE=1  

mkdir ${HOME}/local/scratch/pennant/${SLURM_JOB_ID}
cd ${HOME}/local/scratch/pennant/${SLURM_JOB_ID}
cp ${HOME}/local/src/pennant/pennant . 
cp -r ${HOME}/local/src/pennant/test  . 

echo ${SLURM_JOB_ID}
${HOME}/local/bin/gasnetrun_ibv -n 2  ./pennant -ll:cpu 4 -ll:dma 3 -n 10 -f ./test/sedov/sedov.pnt 


#GASNET_BACKTRACE=1  GASNET_MASTERIP='127.0.0.1' GASNET_SPAWN=-L SSH_SERVERS="localhost" amudprun -np 1 $@ 

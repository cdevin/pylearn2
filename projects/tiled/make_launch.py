base = """jobdispatch --gpu --pre_tasks='export THEANO_FLAGS=floatX=float32,device=gpu${BLOCK_TASK_ID},force_device=True,base_compiledir=/localscratch/$USER/.theano/' --pre_tasks='rm -rf /tmp/$USER' /home/mmirza/projects/noisylearn/projects/tiled/worker.sh %(path)s"{{%(args)s}}\""""
args = ','.join([str(job_id) for job_id in xrange(20)])
#path = "/gs/scratch/mmirza/results/pentree_dense/"
#path = "/gs/scratch/mmirza/results/pentree_sparse_local1/"
#path = "/gs/scratch/mmirza/results/pentree_sparse_local_composite/"
#path = "/gs/scratch/mmirza/results/pentree_sparse_local1_2/"
#path = "/gs/scratch/mmirza/results//pentree_sparse_local1_rec/"
path = "/gs/scratch/mmirza/results/pentree_sparse_relu_composite/"
f = open('launch.sh', 'w')
f.write(base % locals())
f.close()

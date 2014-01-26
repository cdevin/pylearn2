base = """jobdispatch --gpu --pre_tasks='export THEANO_FLAGS=floatX=float32,device=gpu${BLOCK_TASK_ID},force_device=True,base_compiledir=/localscratch/$USER/.theano/' --pre_tasks='rm -rf /tmp/$USER' /home/mmirza/projects/noisylearn/projects/tiled/worker.sh %(path)s"{{%(args)s}}\""""
args = ','.join([str(job_id) for job_id in xrange(30)])
path = "/gs/scratch/mmirza/results/pentree_dense/"
f = open('launch.sh', 'w')
f.write(base % locals())
f.close()

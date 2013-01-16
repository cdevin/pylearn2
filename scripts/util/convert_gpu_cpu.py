import numpy
from pylearn2.utils import serial


"""
Conver project models' shared varibales from gpu arrays to cpu arrays
"""

def siamese_mix_single_category(cpu, gpu, save_path):

    cpu = serial.load(cpu)
    gpu = serial.load(gpu)

    cpu_params = cpu.conv._params + cpu.conv_mlp._params + cpu.mlp._params
    gpu_params = gpu.conv._params + gpu.conv_mlp._params + gpu.mlp._params

    for cpu_p, gpu_p in zip(cpu_params, gpu_params):
        cpu_p.set_value(gpu_p.get_value(borrow= True),borrow = True)

    serial.save(save_path, cpu)


def test_siamese_mix_single_category(cpu, gpu):
    cpu = serial.load(cpu)
    gpu = serial.load(gpu)

    cpu_params = cpu.conv._params + cpu.conv_mlp._params + cpu.mlp._params
    gpu_params = gpu.conv._params + gpu.conv_mlp._params + gpu.mlp._params

    for cpu_p, gpu_p in zip(cpu_params, gpu_params):
        assert numpy.array_equal(cpu_p.get_value(), gpu_p.get_value())

if __name__ == "__main__":
    cpu = '/data/lisatmp/mirzamom/tmp/best/siamese_mix_cpu.pkl'
    gpu = '/data/lisatmp/mirzamom/tmp/best/siamese_mix_gpu.pkl'
    save_path = '/data/lisatmp/mirzamom/tmp/best/siamese_mix_cpu_correct.pkl'
    print 'converting'
    siamese_mix_single_category(cpu, gpu, save_path)
    print 'testing'
    test_siamese_mix_single_category(save_path, gpu)

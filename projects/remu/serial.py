import os
import shutil
from pylearn2.utils import serial

def copy_load(src, dest = "WORK_DIR", file_name = None):
    """
    Copy the data to a dest and the loads it.
    This is usefull to fist move the data to loca disc

    src: path to source file
    dest: path to destination. If set to WORK_DIR,
        will copy to current working dir.
    """

    if dest == "WORK_DIR":
        assert file_name is not None
        dest = os.path.join(os.getcwd(), file_name)

    shutil.copyfile(src, dest)
    return serial.load(dest)

"""Compiled batch renderer for already-screened, complete positive-price inputs."""
from pathlib import Path
import ctypes,hashlib,subprocess
import numpy as np


def load_renderer(build_dir):
    source=Path(__file__).with_name('fast_renderer.c');tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True);library=build_dir/f'render_{tag}.dylib'
    if not library.exists():
        tmp=library.with_suffix('.tmp.dylib')
        subprocess.run(['clang','-O3','-shared','-fPIC',str(source),'-o',str(tmp)],check=True);tmp.replace(library)
    lib=ctypes.CDLL(str(library));f=lib.render
    arr=np.ctypeslib.ndpointer
    f.argtypes=[arr(np.float64,flags='C_CONTIGUOUS'),arr(np.float64,flags='C_CONTIGUOUS'),arr(np.int64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ctypes.c_int,arr(np.uint8,flags='C_CONTIGUOUS')];f.restype=ctypes.c_int
    def render(ohlcv,tri,ends,window=20,ma=20):
        ohlcv=np.ascontiguousarray(ohlcv,dtype='float64');tri=np.ascontiguousarray(tri,dtype='float64');ends=np.ascontiguousarray(ends,dtype='int64')
        if window<1 or ma<1 or ohlcv.ndim!=2 or ohlcv.shape!=(len(tri),5):raise ValueError('Invalid renderer arguments')
        if len(ends) and (ends.min()<window+ma-2 or ends.max()>=len(tri)):raise ValueError('Incomplete renderer history')
        out=np.zeros((len(ends),1,64,3*window),dtype='uint8')
        failure=f(ohlcv,tri,ends,len(ends),window,ma,out)
        if failure:raise ValueError(f'Invalid normalized prices in image {failure-1}')
        return out
    return render

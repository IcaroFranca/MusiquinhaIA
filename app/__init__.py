import os

# Antes de importar numpy/torch: o OpenBLAS reserva buffers por thread e, com várias
# análises em paralelo em 16 threads, a memória virtual do Windows esgota.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

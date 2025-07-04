# libgemm
CUDA cuBLAS DGEMM and SGEMM, Ozaki Scheme I EF and Ozaki Scheme II (Fast and Accurate Modes) Interception Library for BLAS dgemm and CBLAS cblas_dgemm

## Usage
### Compiling
```bash
bash compile.sh
```
### Using
```bash
LD_PRELOAD=/path/to/libgemm/lib/libgemm.so python <script>
```

libgemm uses the ```LIBGEMM_OP_MODE``` to determine the interception redirection behaviour

Valid Modes:
* 0 - Netlib BLAS dgemm_ / Netlib CBLAS cblas_dgemm
* 10 - cuBLAS dgemm
* 15 - cuBLAS sgemm
* 103-117 - Ozaki Scheme I EF with Splits 3-17
* 202-220 - Ozaki Scheme II Fast Mode with Moduli 2-20
* 302-320 - Ozaki Scheme II Accurate Mode with Moduli 2-20

### PySCF bindings

Patched ```numpy_helper.py``` file to enable interception of only ```lib.dgemm``` and ```lib.einsum``` calls for PySCF.

Requires the ```LIBGEMM_LIMITED_OP``` environment variable as an intercept target. The patched script will set and reset the ```LIBGEMM_OP_MODE``` target for libgemm based on this variable. Valid operation modes are the same as the ```LIBGEMM_OP_MODE``` modes.

It is recommended to compile the C backend for numpy_helper with the No-OpenMP flag, since parallel dgemm calls cause issues with the Ozaki Scheme II code. In the pyscf/lib/np_helper directory, run:
```bash
gcc *.c -I.. -O3 -shared -fPIC -fno-openmp -o ../libnp_helper.so
```

### Debug Mode
```libgemm_debug.so``` computes the L2 error between the OS-II matrices and cuBLAS dgemm matrices as reference. 

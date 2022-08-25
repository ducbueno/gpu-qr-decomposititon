__kernel void up_trsolve(const unsigned int nbcols,
                         const unsigned int bs,
                         const unsigned int br,
                         __global const double *mat,
                         __global const double *B,
                         __global double *X)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    X[i * bs + j] = B[i * bs + j] / mat[dense_block_ind(nbcols, bs, br, br, i, i)];

    for(unsigned int k = 1; k < bs; k++){
        if(i < k){
            X[i * bs + j] -= mat[dense_block_ind(nbcols, bs, br, br, i, k)] * X[k * bs + j] / mat[dense_block_ind(nbcols, bs, br, br, i, i)];
        }
    }
}

__kernel void up_trsolve_mat(const unsigned int nbcols,
                             const unsigned int bs,
                             const unsigned int br,
                             const unsigned int bc,
                             __global double *mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    mat[dense_block_ind(nbcols, bs, br, bc, i, j)] /= mat[dense_block_ind(nbcols, bs, br, br, i, i)];

    for(unsigned int k = 1; k < bs; k++){
        if(i < k){
            mat[dense_block_ind(nbcols, bs, br, bc, i, j)] -= mat[dense_block_ind(nbcols, bs, br, br, i, k)] * \
                mat[dense_block_ind(nbcols, bs, br, bc, k, j)] / mat[dense_block_ind(nbcols, bs, br, br, i, i)];
        }
    }
}

__kernel void block_mult_sub(const unsigned int nbcols,
                             const unsigned int bs,
                             const unsigned int br,
                             const unsigned int bc,
                             __global const double *mat,
                             __global const double *B,
                             __global double *C)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    for(unsigned int k = 0; k < bs; k++){
        C[i * bs + j] -= mat[dense_block_ind(nbcols, bs, br, bc, i, k)] * B[k * bs + j];
    }
}

__kernel void solve(const unsigned int nbcols,
                    const unsigned int bs,
                    __global double *mat,
                    __global const double *b,
                    __global double *x)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;

    if(lane < num_active_threads){
        for(unsigned int br = lane / bs / bs; br < nbcols; br += num_rows_per_warp){
            up_trsolve(nbcols, bs, br, mat, b + br * bs * bs, x + br * bs * bs);
        }

        for(unsigned int _br = 1; _br < nbcols; _br++){
            for(unsigned int br = lane / bs / bs; br < nbcols - _br; br += num_rows_per_warp){
                up_trsolve_mat(nbcols, bs, br, nbcols - _br, mat);
                block_mult_sub(nbcols, bs, br, nbcols - _br, mat, x + (nbcols - _br) * bs * bs, x + br * bs * bs);
            }
        }
    }
}

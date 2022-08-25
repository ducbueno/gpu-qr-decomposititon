// This file is auto-generated. Do not edit!

#include "OpenclKernels.hpp"

const std::string OpenclKernels::qr_decomposition_str = R"( 
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

void atomic_add(__local double *val, double delta)
{
    union{
        double f;
        ulong i;
    } old;
    union {
        double f;
        ulong i;
    } new;

    do{
        old.f = *val;
        new.f = old.f + delta;
    } while(atom_cmpxchg((volatile __local ulong *)val, old.i, new.i) != old.i);
}

unsigned int dense_block_ind(const unsigned int nbcols,
                             const unsigned int bs,
                             const unsigned int br,
                             const unsigned int bc,
                             const unsigned int r,
                             const unsigned int c)
{
    return nbcols * br * bs * bs + (r * nbcols + bc) * bs + c;
}

__kernel void sp2dense(const unsigned int nbrows,
                       const unsigned int nbcols,
                       const unsigned int bs,
                       __global const int *ptrs,
                       __global const int *inds,
                       __global const double *vals,
                       __global double *rv_mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int c = (lane / bs) % bs;
    const unsigned int r = lane % bs;
    unsigned int br = lane / bs / bs;

    if(lane < num_active_threads){
        while(br < nbrows){
            for(unsigned int ptr = ptrs[br]; ptr < ptrs[br + 1]; ptr++){
                rv_mat[dense_block_ind(nbcols, bs, br, inds[ptr], r, c)] = vals[ptr * bs * bs + r * bs + c];
            }

            br += num_rows_per_warp;
        }
    }
}

__kernel void set_qx0(const unsigned int bs,
                      const unsigned int eye_idx,
                      __global double *qx0)
{
    const unsigned int idx_t = get_local_id(0);

    if(idx_t < bs){
        qx0[eye_idx * bs * bs + idx_t * bs + idx_t] = 1.0;
    }
}

__kernel void coldotp(const unsigned int nbrows,
                      const unsigned int nbcols,
                      const unsigned int bs,
                      const unsigned int coll,
                      const unsigned int colr,
                      const unsigned int row_offset,
                      __global const double *mat,
                      __local double *sum)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    sum[lane] = 0.0;

    for(unsigned int r = lane + coll + row_offset; r < nbrows * bs; r += warpsize){
        sum[lane] += mat[r * nbcols * bs + coll] * mat[r * nbcols * bs + colr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int stride = warpsize / 2; stride > 0; stride /= 2){
        if(lane < stride){
            sum[lane] += sum[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void sub_scale_col(const unsigned int nbrows,
                            const unsigned int nbcols,
                            const unsigned int bs,
                            const unsigned int coll,
                            const unsigned int colr,
                            const unsigned int row_offset,
                            const double factor,
                            __global double *mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    for(unsigned int r = lane + colr + row_offset; r < nbrows * bs; r += warpsize){
        mat[r * nbcols * bs + coll] -= factor * mat[r * nbcols * bs + colr];
    }
}

__kernel void scale_col(const unsigned int nbrows,
                        const unsigned int nbcols,
                        const unsigned int bs,
                        const unsigned int col,
                        const unsigned int row_offset,
                        const double factor,
                        __global double *mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    for(unsigned int r = lane + col + row_offset; r < nbrows * bs; r += warpsize){
        mat[r * nbcols * bs + col] /= factor;
    }
}

__kernel void tile_house(const unsigned int nbrows,
                         const unsigned int nbcols,
                         const unsigned int bs,
                         const unsigned int tile,
                         __global double *mat,
                         __local double *sum,
                         __local double *T)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    double v0, beta, mu;

    for(unsigned int col = tile * bs; col < (tile + 1) * bs; col++){
        coldotp(nbrows, nbcols, bs, col, col, 1, mat, sum);

        double alpha = mat[col * nbcols * bs + col];
        double sigma = sum[0];

        if(sigma == 0){
            v0 = 1.0;
            beta = alpha >= 0 ? 0.0 : -2.0;
        }
        else{
            mu = sqrt(alpha * alpha + sigma);
            v0 = alpha <= 0 ? alpha - mu : -sigma / (alpha + mu);
            beta = 2 * (v0 * v0) / (sigma + v0 * v0);
        }

        for(unsigned int i = 0; i < bs - (col - tile * bs); i++){
            unsigned int _col = (tile + 1) * bs - i - 1;

            coldotp(nbrows, nbcols, bs, col, _col, 1, mat, sum);
            alpha = mat[col * nbcols * bs + _col];
            double s = alpha + sum[0] / v0;
            mat[col * nbcols * bs + _col] -= beta * s;

            if(_col > col){
                sub_scale_col(nbrows, nbcols, bs, _col, col, 1, beta * s / v0, mat);
            }
        }

        scale_col(nbrows, nbcols, bs, col, 1, v0, mat);
    }

    if(lane < bs * bs){
        T[lane] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int i = 0; i < bs; i++){
        coldotp(nbrows, nbcols, bs, tile * bs + i, tile * bs + i, 1, mat, sum);
        T[i * bs + i] = (1 + sum[0]) / 2;

        for(unsigned int j = i + 1; j < bs; j++){
            coldotp(nbrows, nbcols, bs, tile * bs + i, tile * bs + j, j + 1, mat, sum);
            T[i * bs + j] = mat[(tile * bs + j) * nbcols * bs + (tile * bs + i)] + sum[0];
        }
    }
}

__kernel void block_coldotp_transp(const unsigned int nbrows,
                                   const unsigned int nbcols,
                                   const unsigned int bs,
                                   const unsigned int bc,
                                   const unsigned int tile,
                                   __global const double *mat,
                                   __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_cols_per_warp = warpsize / bs / bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    W[lane] = 0.0;

    for(unsigned int _bc = 0; _bc < num_cols_per_warp && bc + _bc < nbcols; _bc++){
        if(lane < num_active_threads){
            for(unsigned int br = tile + lane / bs / bs; br < nbrows; br += num_rows_per_warp){
                double temp = 0.0;

                for(unsigned int k = 0; k < bs; k++){
                    if(br == tile){
                        if(k == 0){
                            temp += mat[dense_block_ind(nbcols, bs, br, bc + _bc, i, j)];
                        }
                        else if(k > i){
                            temp += mat[dense_block_ind(nbcols, bs, br, tile, k, i)] * \
                                mat[dense_block_ind(nbcols, bs, br, bc + _bc, k, j)];
                        }
                    }
                    else{
                        temp += mat[dense_block_ind(nbcols, bs, br, tile, k, i)] * \
                            mat[dense_block_ind(nbcols, bs, br, bc + _bc, k, j)];
                    }
                }

                atomic_add(W + _bc * bs * bs + i * bs + j, temp);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void block_coldotp_transp_qx(const unsigned int nbrows,
                                      const unsigned int nbcols,
                                      const unsigned int bs,
                                      const unsigned int tile,
                                      __global const double *mat,
                                      __global const double *qx,
                                      __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    W[lane] = 0.0;

    if(lane < num_active_threads){
        for(unsigned int br = tile + lane / bs / bs; br < nbrows; br += num_rows_per_warp){
            double temp = 0.0;

            for(unsigned int k = 0; k < bs; k++){
                if(br == tile){
                    if(k == 0){
                        temp += qx[br * bs * bs + i * bs + j];
                    }
                    else if(k > i){
                        temp += mat[dense_block_ind(nbcols, bs, br, tile, k, i)] * \
                            qx[br * bs * bs + k * bs + j];
                    }
                }
                else{
                    temp += mat[dense_block_ind(nbcols, bs, br, tile, k, i)] * \
                        qx[br * bs * bs + k * bs + j];
                }
            }

            atomic_add(W + i * bs + j, temp);
        }
    }
}

__kernel void block_col_trsolve(const unsigned int nbrows,
                                const unsigned int nbcols,
                                const unsigned int bs,
                                __local const double *T,
                                __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    if(lane < num_active_threads){
        unsigned int blk = lane / bs / bs;

        W[blk * bs * bs + i * bs + j] /= T[i * bs + i];

        for(unsigned int k = 0; k < bs - 1; k++){
            if(i > k){
                W[blk * bs * bs + i * bs + j] -= T[k * bs + i] * W[blk * bs * bs + k * bs + j] / T[i * bs + i];
            }
        }
    }
}

__kernel void block_col_mult_sub(const unsigned int nbrows,
                                 const unsigned int nbcols,
                                 const unsigned int bs,
                                 const unsigned int bc,
                                 const unsigned int tile,
                                 __global double *mat,
                                 __local const double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_cols_per_warp = warpsize / bs / bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    for(unsigned int _bc = 0; _bc < num_cols_per_warp && bc + _bc < nbcols; _bc++){
        if(lane < num_active_threads){
            for(unsigned int br = tile + lane / bs / bs; br < nbrows; br += num_rows_per_warp){
                for(unsigned int k = 0; k < bs; k++){
                    if(br == tile){
                        if(k == 0){
                            mat[dense_block_ind(nbcols, bs, br, bc + _bc, i, j)] -= W[_bc * bs * bs + i * bs + j];
                        }
                        else if(k < i){
                            mat[dense_block_ind(nbcols, bs, br, bc + _bc, i, j)] -= mat[dense_block_ind(nbcols, bs, br, tile, i, k)] * \
                                W[_bc * bs * bs + k * bs + j];
                        }
                    }
                    else{
                        mat[dense_block_ind(nbcols, bs, br, bc + _bc, i, j)] -= mat[dense_block_ind(nbcols, bs, br, tile, i, k)] * \
                            W[_bc * bs * bs + k * bs + j];
                    }
                }
            }
        }
    }
}

__kernel void block_col_mult_sub_qx(const unsigned int nbrows,
                                    const unsigned int nbcols,
                                    const unsigned int bs,
                                    const unsigned int tile,
                                    __global const double *mat,
                                    __global double *qx,
                                    __local const double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    if(lane < num_active_threads){
        for(unsigned int br = tile + lane / bs / bs; br < nbrows; br += num_rows_per_warp){
            for(unsigned int k = 0; k < bs; k++){
                if(br == tile){
                    if(k == 0){
                        qx[br * bs * bs + i * bs + j] -= W[i * bs + j];
                    }
                    else if(k < i){
                        qx[br * bs * bs + i * bs + j] -= mat[dense_block_ind(nbcols, bs, br, tile, i, k)] * W[k * bs + j];
                    }
                }
                else{
                    qx[br * bs * bs + i * bs + j] -= mat[dense_block_ind(nbcols, bs, br, tile, i, k)] * W[k * bs + j];
                }
            }
        }
    }
}

__kernel void update_tr(const unsigned int nbrows,
                        const unsigned int nbcols,
                        const unsigned int bs,
                        const unsigned int tile,
                        __global double *mat,
                        __local const double *T,
                        __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int num_cols_per_warp = warpsize / bs / bs;

    for(unsigned int bc = tile + 1; bc < nbcols; bc += num_cols_per_warp){
        block_coldotp_transp(nbrows, nbcols, bs, bc, tile, mat, W);
        block_col_trsolve(nbrows, nbcols, bs, T, W);
        block_col_mult_sub(nbrows, nbcols, bs, bc, tile, mat, W);
    }
}

__kernel void update_qx(const unsigned int nbrows,
                        const unsigned int nbcols,
                        const unsigned int bs,
                        const unsigned int tile,
                        __global const double *mat,
                        __global double *qx,
                        __local const double *T,
                        __local double *W)
{
    unsigned int lane = get_local_id(0);
    block_coldotp_transp_qx(nbrows, nbcols, bs, tile, mat, qx, W);
    block_col_trsolve(nbrows, nbcols, bs, T, W);
    block_col_mult_sub_qx(nbrows, nbcols, bs, tile, mat, qx, W);
}

__kernel void qr_decomposition(const unsigned int nbrows,
                               const unsigned int nbcols,
                               const unsigned int tile,
                               const unsigned int bs,
                               const unsigned int eye_idx,
                               __global const int *ptrs,
                               __global const int *inds,
                               __global const double *vals,
                               __global double *mat,
                               __global double *qx,
                               __local double *aux)
{
    if(tile == 0){
        sp2dense(nbrows, nbcols, bs, ptrs, inds, vals, mat);
        set_qx0(bs, eye_idx, qx);
    }

    __local double T[9];
    tile_house(nbrows, nbcols, bs, tile, mat, aux, T);
    update_tr(nbrows, nbcols, bs, tile, mat, T, aux);
    update_qx(nbrows, nbcols, bs, tile, mat, qx, T, aux);
}
)"; 

const std::string OpenclKernels::solve_str = R"( 
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
)"; 


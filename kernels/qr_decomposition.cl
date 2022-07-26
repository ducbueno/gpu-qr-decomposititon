unsigned int dense_block_ind(const unsigned int nbcols,
                             const unsigned int block_size,
                             const unsigned int br,
                             const unsigned int bc,
                             const unsigned int r,
                             const unsigned int c)
{
    const unsigned int bs = block_size;
    return nbcols * br * bs * bs + (r * nbcols + bc) * bs + c;
}

__kernel void sp2dense(unsigned int nbrows,
                       unsigned int nbcols,
                       unsigned int block_size,
                       __global const int *ptrs,
                       __global const int *inds,
                       __global const double *vals,
                       __global double *rv_mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
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

__kernel void coldotp(unsigned int nbrows,
                      unsigned int nbcols,
                      unsigned int block_size,
                      unsigned int coll,
                      unsigned int colr,
                      unsigned int row_offset,
                      __global double *mat,
                      __local double *sum)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
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

__kernel void sub_scale_col(unsigned int nbrows,
                            unsigned int nbcols,
                            unsigned int block_size,
                            unsigned int coll,
                            unsigned int colr,
                            unsigned int row_offset,
                            double factor,
                            __global double *mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
    const unsigned int lane = idx_t % warpsize;

    for(unsigned int r = lane + colr + row_offset; r < nbrows * bs; r += warpsize){
        mat[r * nbcols * bs + coll] -= factor * mat[r * nbcols * bs + colr];
    }
}

__kernel void scale_col(unsigned int nbrows,
                        unsigned int nbcols,
                        unsigned int block_size,
                        unsigned int col,
                        unsigned int row_offset,
                        double factor,
                        __global double *mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
    const unsigned int lane = idx_t % warpsize;

    for(unsigned int r = lane + col + row_offset; r < nbrows * bs; r += warpsize){
        mat[r * nbcols * bs + col] /= factor;
    }
}

__kernel void tile_house(unsigned int nbrows,
                         unsigned int nbcols,
                         unsigned int block_size,
                         unsigned int tile,
                         __global double *mat,
                         __local double *sum,
                         __local double *T)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
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

        for(unsigned int i = 0; i < bs - col; i++){
            unsigned int _col = bs - i - 1;

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

    /* if(lane < bs * bs){ */
    /*     T[lane] = 0.0; */
    /* } */
    /* barrier(CLK_LOCAL_MEM_FENCE); */

    /* for(unsigned int i = tile * bs; i < (tile + 1) * bs; i++){ */
    /*     coldotp(nbrows, nbcols, bs, i, i, 1, mat, sum); */
    /*     T[i * bs + i] = (1 + sum[0]) / 2; */

    /*     for(unsigned int j = i + 1; j < (tile + 1) * bs; j++){ */
    /*         coldotp(nbrows, nbcols, bs, i, j, j + 1, mat, sum); */
    /*         T[i * bs + j] = sum[0]; */
    /*     } */
    /* } */
}

__kernel void qr_decomposition(unsigned int nbrows,
                               unsigned int nbcols,
                               unsigned int block_size,
                               __global const int *ptrs,
                               __global const int *inds,
                               __global const double *vals,
                               __global double *rv_mat,
                               __local double *aux)
{
    __local double T[9];

    sp2dense(nbrows, nbcols, block_size, ptrs, inds, vals, rv_mat);

    for(unsigned int tile = 0; tile < 1; tile++){
        tile_house(nbrows, nbcols, block_size, tile, rv_mat, aux, T);
        /* larft(panel, nbcols, nbrows, block_size, rv_mat, aux, t_mat); */
        /* larfb(panel, nbcols, nbrows, block_size, rv_mat, aux, t_mat); */
    }
}

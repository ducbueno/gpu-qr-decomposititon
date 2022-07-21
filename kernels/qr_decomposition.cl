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

__kernel void sp_to_dense(unsigned int nbrows,
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
                rv_mat[dense_block_ind(nbrows, bs, inds[ptr], br, r, c)] = vals[ptr * bs * bs + r * bs + c];
            }

            br += num_rows_per_warp;
        }
    }
}

__kernel void panel_qr(unsigned int panel,
                       unsigned int nbrows,
                       unsigned int nbcols,
                       unsigned int block_size,
                       __global double *rv_mat,
                       __local double *sum)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
    const unsigned int lane = idx_t % warpsize;
    double sigma[3], rv_row[3], rho[3];

    for(unsigned int c = 0; c < bs; c++){
        for(unsigned int _c = c; _c < bs; _c++){
            sum[lane] = 0.0;

            for(unsigned int r = panel * bs + lane + c + 1; r < nbrows * bs; r += warpsize){
                sum[lane] += rv_mat[(r * nbcols + panel) * bs + c] * rv_mat[(r * nbcols + panel) * bs + _c];

                for(unsigned int stride = warpsize / 2; stride > 0; stride /= 2){
                    barrier(CLK_LOCAL_MEM_FENCE);

                    if(lane < stride){
                        sum[lane] += sum[lane + stride];
                    }
                }
            }

            sigma[_c] = sum[0];
            rv_row[_c] = rv_mat[dense_block_ind(nbcols, bs, panel, panel, c, _c)];
        }

        double mu = sqrt(rv_row[c] + sigma[c]);
        double v0 = (rv_row[c] <= 0) ? rv_row[c] - mu : -sigma[c] / (rv_row[c] + mu);
        double beta = 2 * v0 * v0 / (sigma[c] + v0 * v0);

        for(unsigned int _c = c; _c < bs; _c++){
            rho[_c] = beta * (rv_row[_c] + sigma[_c] / v0);
            rv_mat[dense_block_ind(nbcols, bs, panel, panel, c, _c)] = rv_row[_c] - rho[_c];

            for(unsigned int r = panel * bs + lane + c + 1; r < nbrows * bs; r += warpsize){
                if(_c == c){
                    rv_mat[(r * nbcols + panel) * bs + _c] /= v0;
                }
                else{
                    rv_mat[(r * nbcols + panel) * bs + _c] -= rho[_c] * rv_mat[(r * nbcols + panel) * bs + c];
                }
            }
        }
    }
}

__kernel void larft(unsigned int panel,
                    unsigned int nbrows,
                    unsigned int nbcols,
                    unsigned int block_size,
                    __global double *rv_mat,
                    __local double *sum,
                    __local double *t_mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
    const unsigned int lane = idx_t % warpsize;

    if(lane < bs * bs){
        t_mat[lane] = 0.0;
    }

    for(int c = 0; c < bs; c++){
        for(int _c = c; _c >= 0; _c--){
            sum[lane] = 0.0;

            for(unsigned int r = panel * bs + lane + c + 1; r < nbrows * bs; r += warpsize){
                sum[lane] += rv_mat[(r * nbcols + panel) * bs + c] * rv_mat[(r * nbcols + panel) * bs + _c];

                for(unsigned int stride = warpsize / 2; stride > 0; stride /= 2){
                    if(lane < stride){
                        sum[lane] += sum[lane + stride];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }

            if(lane == 0){
                t_mat[_c * bs + c] = sum[0];
                t_mat[_c * bs + c] += (_c == c) ? 1 : rv_mat[(c * nbcols + panel) * bs + _c];
            }
        }
    }

    if(lane < bs * bs && panel == 0){
        printf("t_mat[%d] = %f\n", lane, t_mat[lane]);
    }
}

__kernel void gemm1(unsigned int nbcols,
                    unsigned int br,
                    unsigned int bc,
                    unsigned int panel,
                    unsigned int result_ind,
                    __global double *rv_mat,
                    __local double *result)
{
    const unsigned int bs = 3;
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int r = lane % bs;
    const unsigned int c = (lane / bs) % bs;
    double temp = 0.0;

    for(unsigned int k = 0; k < bs; k++){
        if(br == panel){ // rv_mat[panel, panel] is unit lower triagular
            if(k == r){
                temp += rv_mat[dense_block_ind(nbcols, bs, br, bc, r, c)];
            }
            else if(k > r){
                temp += rv_mat[dense_block_ind(nbcols, bs, br, panel, k, r)] * \
                    rv_mat[dense_block_ind(nbcols, bs, br, bc, k, c)];
            }
        }
        else{
            temp += rv_mat[dense_block_ind(nbcols, bs, br, panel, k, r)] * \
                rv_mat[dense_block_ind(nbcols, bs, br, bc, k, c)];
        }
    }

    result[result_ind * bs * bs + r * bs + c] += temp;
}

__kernel void trmm(unsigned int result_ind,
                   __local double *result,
                   __local double *t_mat)
{
    const unsigned int bs = 3;
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int r = lane % bs;
    const unsigned int c = (lane / bs) % bs;
    double temp = 0.0;

    for(unsigned int k = 0; k < bs; k++){
        temp += t_mat[k * bs + r] * result[result_ind * bs * bs + k * bs + c];
    }

    result[result_ind * bs * bs + r * bs + c] = temp;
}

__kernel void gemm2(unsigned int nbcols,
                    unsigned int br,
                    unsigned int bc,
                    unsigned int panel,
                    unsigned int result_ind,
                    __global double *rv_mat,
                    __local double *result,
                    __local double *t_mat)
{
    const unsigned int bs = 3;
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int r = lane % bs;
    const unsigned int c = (lane / bs) % bs;
    double temp = 0.0;

    if(br == panel){
        for(unsigned int k = 0; k < bs; k++){
            temp += t_mat[k * bs + r] * result[result_ind * bs * bs + k * bs + c];
        }

        result[result_ind * bs * bs + r * bs + c] = temp;
        temp = 0.0;
    }

    for(unsigned int k = 0; k < bs; k++){
        if(br == panel){ // rv_mat[panel, panel] is unit lower triagular
            if(k == r){
                temp += result[result_ind * bs * bs + r * bs + c];
            }
            else if(k < r){
                temp += rv_mat[dense_block_ind(nbcols, bs, br, panel, r, k)] * \
                    result[result_ind * bs * bs + k * bs + c];
            }
        }
        else{
            temp += rv_mat[dense_block_ind(nbcols, bs, br, panel, r, k)] * \
                result[result_ind * bs * bs + k * bs + c];
        }
    }

    rv_mat[dense_block_ind(nbcols, bs, br, bc, r, c)] -= temp;
}

__kernel void larfb(unsigned int panel,
                    unsigned int nbrows,
                    unsigned int nbcols,
                    unsigned int block_size,
                    __global double *rv_mat,
                    __local double *vta2,
                    __local double *t_mat)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_cols_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    unsigned int bc = 1 + panel + lane / bs / bs;

    if(lane < num_active_threads){
        while(bc < nbcols){
            vta2[lane] = 0.0;

            for(unsigned int br = panel; br < nbrows; br++){
                gemm1(nbcols, br, bc, panel, lane / bs / bs, rv_mat, vta2);
            }

            for(unsigned int br = panel; br < nbrows; br++){
                gemm2(nbcols, br, bc, panel, lane / bs / bs, rv_mat, vta2, t_mat);
            }

            bc += num_cols_per_warp;
        }
    }
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
    __local double t_mat[9];

    sp_to_dense(nbrows, block_size, ptrs, inds, vals, rv_mat);

    /* for(unsigned int row = 0; row < nbrows; row++){ */
    for(unsigned int panel = 0; panel < 1; panel++){
        panel_qr(panel, nbcols, nbrows, block_size, rv_mat, aux); // nbcols and nbrows are switched because rv_mat has the shape of the transpose of the shadow matrix
        /* larft(panel, nbcols, nbrows, block_size, rv_mat, aux, t_mat); */
        /* larfb(panel, nbcols, nbrows, block_size, rv_mat, aux, t_mat); */
    }
}

#include "dli.h"

// FIXME(Step 1): accept device containers
void update_hx(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx,
               thrust::device_vector<float> &ez, thrust::device_vector<float> &buffer) {
  
  // FIXME(Step 1): compute transformation on GPU
  // Calculate Ez difference into buffer
  thrust::transform(ez.begin() + n, ez.end(), ez.begin(), buffer.begin(),
                    [=] __device__ (float x, float y) { return x - y; });

  // FIXME(Step 1): compute transformation on GPU
  // Update Hx using buffer
  thrust::transform(hx.begin(), hx.end() - n, buffer.begin(), hx.begin(),
                    [=] __device__ (float h, float cex) {
                      return h - dli::C0 * dt / 1.3f * cex / dy;
                    });
}

// FIXME(Step 1): accept device containers
void update_hy(int n, float dx, float dy, float dt, thrust::device_vector<float> &hy,
               thrust::device_vector<float> &ez, thrust::device_vector<float> &buffer) {
  
  // FIXME(Step 1): compute transformation on GPU
  thrust::transform(ez.begin(), ez.end() - 1, ez.begin() + 1, buffer.begin(),
                    [=] __device__ (float x, float y) { return x - y; });

  // FIXME(Step 1): compute transformation on GPU
  thrust::transform(hy.begin(), hy.end() - 1, buffer.begin(), hy.begin(),
                    [=] __device__ (float h, float cey) {
                      return h - dli::C0 * dt / 1.3f * cey / dx;
                    });
}

// FIXME(Step 1): accept device containers
void update_dz(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx,
               thrust::device_vector<float> &hy, thrust::device_vector<float> &dz,
               thrust::device_vector<int> &cell_ids) {
  
  // Get raw pointers for device lambda to avoid capturing full vectors
  float* raw_hx = thrust::raw_pointer_cast(hx.data());
  float* raw_hy = thrust::raw_pointer_cast(hy.data());
  float* raw_dz = thrust::raw_pointer_cast(dz.data());

  // FIXME(Step 1): compute for each on GPU
  thrust::for_each(cell_ids.begin(), cell_ids.end(),
                   [=] __device__ (int cell_id) {
                     if (cell_id > n) {
                       float hx_diff = raw_hx[cell_id - n] - raw_hx[cell_id];
                       float hy_diff = raw_hy[cell_id] - raw_hy[cell_id - 1];
                       raw_dz[cell_id] += dli::C0 * dt * (hx_diff / dx + hy_diff / dy);
                     }
                   });
}

// FIXME(Step 1): accept device containers
void update_ez(thrust::device_vector<float> &ez, thrust::device_vector<float> &dz) {
  // FIXME(Step 1): compute transformation on GPU
  thrust::transform(dz.begin(), dz.end(), ez.begin(),
                    [=] __device__ (float d) { return d / 1.3f; });
}

// FIXME(Step 1): remove this function (copy_to_host is no longer needed inside simulate)

// Do not change the signature of this function
void simulate(int cells_along_dimension, float dx, float dy, float dt,
              thrust::device_vector<float> &d_hx,
              thrust::device_vector<float> &d_hy,
              thrust::device_vector<float> &d_dz,
              thrust::device_vector<float> &d_ez) {
  
  // FIXME(Step 1): Remove host copies. We now use the passed device vectors directly.
  
  int cells = cells_along_dimension * cells_along_dimension;

  // Materialize cell indices on Device
  thrust::device_vector<int> cell_ids(cells);
  thrust::sequence(cell_ids.begin(), cell_ids.end());

  // Buffer on Device
  thrust::device_vector<float> buffer(cells);

  for (int step = 0; step < dli::steps; step++) {
    update_hx(cells_along_dimension, dx, dy, dt, d_hx, d_ez, buffer);
    update_hy(cells_along_dimension, dx, dy, dt, d_hy, d_ez, buffer);
    update_dz(cells_along_dimension, dx, dy, dt, d_hx, d_hy, d_dz, cell_ids);
    update_ez(d_ez, d_dz);
  }
}
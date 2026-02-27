#include "dli.h"

// FIXME(Step 2): Use zip/transform iterators to avoid materializing buffer
void update_hx(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx,
               thrust::device_vector<float> &ez, thrust::device_vector<float> &buffer) {
    
    // Create zip iterators to read hx, ez[i+n], and ez[i] simultaneously
    auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(hx.begin(), ez.begin() + n, ez.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(hx.end() - n, ez.end(), ez.end() - n));

    // Perform calculation in one pass without temporary memory
    thrust::for_each(zip_in, zip_end, [dt, dx, dy] __device__ (auto t) {
        float &h = thrust::get<0>(t);      // Reference to hx
        float ez_next = thrust::get<1>(t); // ez[i+n]
        float ez_curr = thrust::get<2>(t); // ez[i]
        
        float cex = ez_next - ez_curr; // Computed on the fly
        h = h - dli::C0 * dt / 1.3f * cex / dy;
    });
}

// FIXME(Step 2): Use zip/transform iterators
void update_hy(int n, float dx, float dy, float dt, thrust::device_vector<float> &hy,
               thrust::device_vector<float> &ez, thrust::device_vector<float> &buffer) {
    
    auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(hy.begin(), ez.begin() + 1, ez.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(hy.end() - 1, ez.end(), ez.end() - 1));

    thrust::for_each(zip_in, zip_end, [dt, dx, dy] __device__ (auto t) {
        float &h = thrust::get<0>(t);
        float ez_next = thrust::get<1>(t);
        float ez_curr = thrust::get<2>(t);
        
        float cey = ez_next - ez_curr; // Computed on the fly
        h = h - dli::C0 * dt / 1.3f * cey / dx;
    });
}

// FIXME(Step 2): Remove cell_ids and use counting iterator
void update_dz(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx,
               thrust::device_vector<float> &hy, thrust::device_vector<float> &dz) {
    
    float* raw_hx = thrust::raw_pointer_cast(hx.data());
    float* raw_hy = thrust::raw_pointer_cast(hy.data());
    float* raw_dz = thrust::raw_pointer_cast(dz.data());

    // Iterate using counting_iterator
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(dz.size()),
                     [n, dx, dy, dt, raw_hx, raw_hy, raw_dz] __device__ (int i) {
                        if (i > n) {
                            float hx_diff = raw_hx[i - n] - raw_hx[i];
                            float hy_diff = raw_hy[i] - raw_hy[i - 1];
                            raw_dz[i] += dli::C0 * dt * (hx_diff / dx + hy_diff / dy);
                        }
                     });
}

void update_ez(thrust::device_vector<float> &ez, thrust::device_vector<float> &dz) {
    thrust::transform(dz.begin(), dz.end(), ez.begin(),
                      [] __device__ (float d) { return d / 1.3f; });
}

void simulate(int cells_along_dimension, float dx, float dy, float dt,
              thrust::device_vector<float> &d_hx,
              thrust::device_vector<float> &d_hy,
              thrust::device_vector<float> &d_dz,
              thrust::device_vector<float> &d_ez) {
    
    // Buffer is no longer needed
    thrust::device_vector<float> buffer; 

    for (int step = 0; step < dli::steps; step++) {
        update_hx(cells_along_dimension, dx, dy, dt, d_hx, d_ez, buffer);
        update_hy(cells_along_dimension, dx, dy, dt, d_hy, d_ez, buffer);
        
        // cell_ids removed from call
        update_dz(cells_along_dimension, dx, dy, dt, d_hx, d_hy, d_dz);
        update_ez(d_ez, d_dz);
    }
}
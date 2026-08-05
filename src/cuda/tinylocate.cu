extern "C" __global__ void tln_conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int output_height,
    int output_width,
    int kernel,
    int stride,
    int padding,
    int groups,
    int activation
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = out_channels * output_height * output_width;
    if (index >= count) return;
    int x = index % output_width;
    int y = (index / output_width) % output_height;
    int oc = index / (output_width * output_height);
    int in_per_group = in_channels / groups;
    int out_per_group = out_channels / groups;
    int input_start = (oc / out_per_group) * in_per_group;
    float sum = bias ? bias[oc] : 0.0f;
    for (int local_ic = 0; local_ic < in_per_group; ++local_ic) {
        int ic = input_start + local_ic;
        for (int ky = 0; ky < kernel; ++ky) {
            int iy = y * stride + ky - padding;
            if (iy < 0 || iy >= input_height) continue;
            for (int kx = 0; kx < kernel; ++kx) {
                int ix = x * stride + kx - padding;
                if (ix < 0 || ix >= input_width) continue;
                int input_index = (ic * input_height + iy) * input_width + ix;
                int weight_index = ((oc * in_per_group + local_ic) * kernel + ky) * kernel + kx;
                sum += input[input_index] * weight[weight_index];
            }
        }
    }
    if (activation == 1) {
        float gate = fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
        sum *= gate;
    } else if (activation == 2) {
        sum = 1.0f / (1.0f + expf(-sum));
    }
    output[index] = sum;
}

extern "C" __global__ void tln_add(const float* left, const float* right, float* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = left[index] + right[index];
}

extern "C" __global__ void tln_channel_mean(
    const float* input, float* output, int channels, int height, int width
) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    int area = height * width;
    float sum = 0.0f;
    for (int index = 0; index < area; ++index) sum += input[channel * area + index];
    output[channel] = sum / area;
}

extern "C" __global__ void tln_channel_multiply(
    const float* input, const float* scale, float* output, int channels, int area
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < channels * area) output[index] = input[index] * scale[index / area];
}

extern "C" __global__ void tln_l2_normalize(float* value, int channels, int area) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= area) return;
    float squared = 1.0e-12f;
    for (int channel = 0; channel < channels; ++channel) {
        float item = value[channel * area + position];
        squared += item * item;
    }
    float inverse = rsqrtf(squared);
    for (int channel = 0; channel < channels; ++channel) value[channel * area + position] *= inverse;
}

extern "C" __global__ void tln_fuse_correlation(
    const float* features, const float* query, float* fused, int channels, int area
) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= area) return;
    float score = 0.0f;
    for (int channel = 0; channel < channels; ++channel) {
        float item = features[channel * area + position];
        fused[channel * area + position] = item;
        score += item * query[channel];
    }
    fused[channels * area + position] = score;
}


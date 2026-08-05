//! Framework-free TinyLocate inference on precompiled CUDA kernels.

#![cfg(target_os = "windows")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::f16;

use crate::tinylocate::{ModelWeights, TensorRecord};

const PTX: &str = include_str!("cuda/tinylocate.ptx");
const BN_EPSILON: f32 = 1.0e-5;

struct Tensor {
    data: CudaSlice<f32>,
    channels: usize,
    height: usize,
    width: usize,
}

type DeviceParameters = HashMap<String, (CudaSlice<f32>, CudaSlice<f32>)>;

struct Ops {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    parameters: RefCell<DeviceParameters>,
}

thread_local! {
    static OPS_CACHE: RefCell<Option<Ops>> = const { RefCell::new(None) };
}

fn driver_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

fn values(record: &TensorRecord) -> Vec<f32> {
    record
        .values
        .iter()
        .map(|bits| f16::from_bits(*bits).to_f32())
        .collect()
}

impl Ops {
    fn new() -> io::Result<Self> {
        let context = CudaContext::new(0).map_err(driver_error)?;
        let stream = context.default_stream();
        let module = context
            .load_module(Ptx::from_src(PTX))
            .map_err(driver_error)?;
        Ok(Self {
            stream,
            module,
            parameters: RefCell::new(HashMap::new()),
        })
    }

    fn has_parameters(&self, key: &str) -> bool {
        self.parameters.borrow().contains_key(key)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        cache_key: &str,
        input: &Tensor,
        weight: &[f32],
        bias: &[f32],
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        activation: i32,
    ) -> io::Result<Tensor> {
        let output_height = (input.height + padding * 2 - kernel) / stride + 1;
        let output_width = (input.width + padding * 2 - kernel) / stride + 1;
        let count = out_channels * output_height * output_width;
        if !self.parameters.borrow().contains_key(cache_key) {
            let weight_device = self.stream.clone_htod(weight).map_err(driver_error)?;
            let bias_device = self.stream.clone_htod(bias).map_err(driver_error)?;
            self.parameters
                .borrow_mut()
                .insert(cache_key.to_owned(), (weight_device, bias_device));
        }
        let parameters = self.parameters.borrow();
        let (weight_device, bias_device) = parameters
            .get(cache_key)
            .expect("inserted immediately above when absent");
        let mut output = self
            .stream
            .alloc_zeros::<f32>(count)
            .map_err(driver_error)?;
        let function = self
            .module
            .load_function("tln_conv2d")
            .map_err(driver_error)?;
        let integers = [
            input.channels as i32,
            input.height as i32,
            input.width as i32,
            out_channels as i32,
            output_height as i32,
            output_width as i32,
            kernel as i32,
            stride as i32,
            padding as i32,
            groups as i32,
            activation,
        ];
        let mut launch = self.stream.launch_builder(&function);
        launch.arg(&input.data);
        launch.arg(weight_device);
        launch.arg(bias_device);
        launch.arg(&mut output);
        for value in &integers {
            launch.arg(value);
        }
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }
            .map_err(driver_error)?;
        Ok(Tensor {
            data: output,
            channels: out_channels,
            height: output_height,
            width: output_width,
        })
    }

    fn add(&self, left: &Tensor, right: &Tensor) -> io::Result<Tensor> {
        if (left.channels, left.height, left.width) != (right.channels, right.height, right.width) {
            return Err(io::Error::other("residual tensor shape mismatch"));
        }
        let count = left.channels * left.height * left.width;
        let mut output = self
            .stream
            .alloc_zeros::<f32>(count)
            .map_err(driver_error)?;
        let function = self.module.load_function("tln_add").map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&function);
        launch.arg(&left.data);
        launch.arg(&right.data);
        launch.arg(&mut output);
        let count_i32 = count as i32;
        launch.arg(&count_i32);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }
            .map_err(driver_error)?;
        Ok(Tensor {
            data: output,
            channels: left.channels,
            height: left.height,
            width: left.width,
        })
    }

    fn channel_mean(&self, input: &Tensor) -> io::Result<Tensor> {
        let mut output = self
            .stream
            .alloc_zeros::<f32>(input.channels)
            .map_err(driver_error)?;
        let function = self
            .module
            .load_function("tln_channel_mean")
            .map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&function);
        launch.arg(&input.data);
        launch.arg(&mut output);
        let dimensions = [
            input.channels as i32,
            input.height as i32,
            input.width as i32,
        ];
        for value in &dimensions {
            launch.arg(value);
        }
        unsafe { launch.launch(LaunchConfig::for_num_elems(input.channels as u32)) }
            .map_err(driver_error)?;
        Ok(Tensor {
            data: output,
            channels: input.channels,
            height: 1,
            width: 1,
        })
    }

    fn channel_multiply(&self, input: &Tensor, scale: &Tensor) -> io::Result<Tensor> {
        let area = input.height * input.width;
        let count = input.channels * area;
        let mut output = self
            .stream
            .alloc_zeros::<f32>(count)
            .map_err(driver_error)?;
        let function = self
            .module
            .load_function("tln_channel_multiply")
            .map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&function);
        launch.arg(&input.data);
        launch.arg(&scale.data);
        launch.arg(&mut output);
        let dimensions = [input.channels as i32, area as i32];
        for value in &dimensions {
            launch.arg(value);
        }
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }
            .map_err(driver_error)?;
        Ok(Tensor {
            data: output,
            channels: input.channels,
            height: input.height,
            width: input.width,
        })
    }

    fn normalize(&self, mut input: Tensor) -> io::Result<Tensor> {
        let area = input.height * input.width;
        let function = self
            .module
            .load_function("tln_l2_normalize")
            .map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&function);
        launch.arg(&mut input.data);
        let dimensions = [input.channels as i32, area as i32];
        for value in &dimensions {
            launch.arg(value);
        }
        unsafe { launch.launch(LaunchConfig::for_num_elems(area as u32)) }.map_err(driver_error)?;
        Ok(input)
    }

    fn fuse(&self, features: &Tensor, query: &Tensor) -> io::Result<Tensor> {
        let area = features.height * features.width;
        let mut output = self
            .stream
            .alloc_zeros::<f32>((features.channels + 1) * area)
            .map_err(driver_error)?;
        let function = self
            .module
            .load_function("tln_fuse_correlation")
            .map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&function);
        launch.arg(&features.data);
        launch.arg(&query.data);
        launch.arg(&mut output);
        let dimensions = [features.channels as i32, area as i32];
        for value in &dimensions {
            launch.arg(value);
        }
        unsafe { launch.launch(LaunchConfig::for_num_elems(area as u32)) }.map_err(driver_error)?;
        Ok(Tensor {
            data: output,
            channels: features.channels + 1,
            height: features.height,
            width: features.width,
        })
    }
}

fn record<'a>(weights: &'a ModelWeights, name: &str) -> io::Result<&'a TensorRecord> {
    weights
        .tensors
        .get(name)
        .ok_or_else(|| io::Error::other(format!("missing TLN tensor {name}")))
}

fn folded(
    weights: &ModelWeights,
    convolution: &str,
    norm: &str,
) -> io::Result<(Vec<f32>, Vec<f32>, usize, usize)> {
    let convolution = record(weights, &format!("{convolution}.weight"))?;
    if convolution.shape.len() != 4 {
        return Err(io::Error::other("convolution weight must be rank four"));
    }
    let out_channels = convolution.shape[0];
    let kernel = convolution.shape[2];
    let gamma = values(record(weights, &format!("{norm}.weight"))?);
    let beta = values(record(weights, &format!("{norm}.bias"))?);
    let mean = values(record(weights, &format!("{norm}.running_mean"))?);
    let variance = values(record(weights, &format!("{norm}.running_var"))?);
    if [gamma.len(), beta.len(), mean.len(), variance.len()]
        .iter()
        .any(|length| *length != out_channels)
    {
        return Err(io::Error::other(
            "batch-normalization tensor shape mismatch",
        ));
    }
    let mut weight = values(convolution);
    let per_channel = weight.len() / out_channels;
    let mut bias = vec![0.0; out_channels];
    for channel in 0..out_channels {
        let scale = gamma[channel] / (variance[channel] + BN_EPSILON).sqrt();
        for item in &mut weight[channel * per_channel..(channel + 1) * per_channel] {
            *item *= scale;
        }
        bias[channel] = beta[channel] - mean[channel] * scale;
    }
    Ok((weight, bias, out_channels, kernel))
}

#[allow(clippy::too_many_arguments)]
fn conv_norm(
    ops: &Ops,
    weights: &ModelWeights,
    input: &Tensor,
    convolution: &str,
    norm: &str,
    stride: usize,
    groups: usize,
    activation: i32,
) -> io::Result<Tensor> {
    let cache_key = format!("{}:{convolution}:folded", weights.fingerprint);
    let convolution_record = record(weights, &format!("{convolution}.weight"))?;
    let out_channels = convolution_record.shape[0];
    let kernel = convolution_record.shape[2];
    let (weight, bias) = if ops.has_parameters(&cache_key) {
        (Vec::new(), Vec::new())
    } else {
        let (weight, bias, _, _) = folded(weights, convolution, norm)?;
        (weight, bias)
    };
    ops.conv(
        &cache_key,
        input,
        &weight,
        &bias,
        out_channels,
        kernel,
        stride,
        kernel / 2,
        groups,
        activation,
    )
}

fn conv_bias(
    ops: &Ops,
    weights: &ModelWeights,
    input: &Tensor,
    prefix: &str,
    activation: i32,
) -> io::Result<Tensor> {
    let weight_record = record(weights, &format!("{prefix}.weight"))?;
    let out_channels = weight_record.shape[0];
    let kernel = weight_record.shape[2];
    let cache_key = format!("{}:{prefix}:bias", weights.fingerprint);
    let (weight, bias) = if ops.has_parameters(&cache_key) {
        (Vec::new(), Vec::new())
    } else {
        (
            values(weight_record),
            values(record(weights, &format!("{prefix}.bias"))?),
        )
    };
    ops.conv(
        &cache_key,
        input,
        &weight,
        &bias,
        out_channels,
        kernel,
        1,
        kernel / 2,
        1,
        activation,
    )
}

fn block(
    ops: &Ops,
    weights: &ModelWeights,
    input: Tensor,
    prefix: &str,
    stride: usize,
    target_channels: usize,
) -> io::Result<Tensor> {
    let source_channels = input.channels;
    let expanded = conv_norm(
        ops,
        weights,
        &input,
        &format!("{prefix}.expand.0"),
        &format!("{prefix}.expand.1"),
        1,
        1,
        1,
    )?;
    let hidden_channels = expanded.channels;
    let depthwise = conv_norm(
        ops,
        weights,
        &expanded,
        &format!("{prefix}.depthwise.0"),
        &format!("{prefix}.depthwise.1"),
        stride,
        hidden_channels,
        1,
    )?;
    let pooled = ops.channel_mean(&depthwise)?;
    let reduced = conv_bias(
        ops,
        weights,
        &pooled,
        &format!("{prefix}.attention.reduce"),
        1,
    )?;
    let scale = conv_bias(
        ops,
        weights,
        &reduced,
        &format!("{prefix}.attention.expand"),
        2,
    )?;
    let attended = ops.channel_multiply(&depthwise, &scale)?;
    let projected = conv_norm(
        ops,
        weights,
        &attended,
        &format!("{prefix}.project.0"),
        &format!("{prefix}.project.1"),
        1,
        1,
        0,
    )?;
    if stride == 1 && source_channels == target_channels {
        ops.add(&input, &projected)
    } else {
        Ok(projected)
    }
}

fn encode(ops: &Ops, weights: &ModelWeights, input: Tensor) -> io::Result<Tensor> {
    let mut value = conv_norm(
        ops,
        weights,
        &input,
        "encoder.stem.0",
        "encoder.stem.1",
        2,
        1,
        1,
    )?;
    for (prefix, stride, channels) in [
        ("encoder.stage1.0", 2, 32),
        ("encoder.stage1.1", 1, 32),
        ("encoder.stage2.0", 2, 64),
        ("encoder.stage2.1", 1, 64),
        ("encoder.stage2.2", 1, 64),
        ("encoder.stage3.0", 1, 96),
        ("encoder.stage3.1", 1, 96),
    ] {
        value = block(ops, weights, value, prefix, stride, channels)?;
    }
    let projection = record(weights, "encoder.project.weight")?;
    let cache_key = format!("{}:encoder.project", weights.fingerprint);
    let (weight, bias) = if ops.has_parameters(&cache_key) {
        (Vec::new(), Vec::new())
    } else {
        (values(projection), vec![0.0; projection.shape[0]])
    };
    let value = ops.conv(
        &cache_key,
        &value,
        &weight,
        &bias,
        projection.shape[0],
        1,
        1,
        0,
        1,
        0,
    )?;
    ops.normalize(value)
}

fn rgb_tensor(ops: &Ops, rgb: &[u8], width: usize, height: usize) -> io::Result<Tensor> {
    if rgb.len() < width * height * 3 {
        return Err(io::Error::other("short RGB input"));
    }
    let area = width * height;
    let mut planar = vec![0.0f32; area * 3];
    for position in 0..area {
        for channel in 0..3 {
            planar[channel * area + position] = rgb[position * 3 + channel] as f32 / 255.0;
        }
    }
    Ok(Tensor {
        data: ops.stream.clone_htod(&planar).map_err(driver_error)?,
        channels: 3,
        height,
        width,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Located {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub confidence: f32,
}

pub fn locate(
    weights: &ModelWeights,
    reference_rgb: &[u8],
    reference_width: usize,
    reference_height: usize,
    search_rgb: &[u8],
    search_width: usize,
    search_height: usize,
) -> io::Result<Located> {
    locate_all(
        weights,
        reference_rgb,
        reference_width,
        reference_height,
        search_rgb,
        search_width,
        search_height,
        0.0,
        1,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| io::Error::other("empty TinyLocate output"))
}

#[allow(clippy::too_many_arguments)]
pub fn locate_all(
    weights: &ModelWeights,
    reference_rgb: &[u8],
    reference_width: usize,
    reference_height: usize,
    search_rgb: &[u8],
    search_width: usize,
    search_height: usize,
    minimum_confidence: f32,
    limit: usize,
) -> io::Result<Vec<Located>> {
    OPS_CACHE.with(|cache| {
        let mut cached = cache
            .try_borrow_mut()
            .map_err(|_| io::Error::other("TinyLocate CUDA runtime was re-entered"))?;
        if cached.is_none() {
            *cached = Some(Ops::new()?);
        }
        locate_all_with_ops(
            cached.as_ref().expect("initialized above"),
            weights,
            reference_rgb,
            reference_width,
            reference_height,
            search_rgb,
            search_width,
            search_height,
            minimum_confidence,
            limit,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn locate_all_with_ops(
    ops: &Ops,
    weights: &ModelWeights,
    reference_rgb: &[u8],
    reference_width: usize,
    reference_height: usize,
    search_rgb: &[u8],
    search_width: usize,
    search_height: usize,
    minimum_confidence: f32,
    limit: usize,
) -> io::Result<Vec<Located>> {
    let reference = encode(
        ops,
        weights,
        rgb_tensor(ops, reference_rgb, reference_width, reference_height)?,
    )?;
    let search = encode(
        ops,
        weights,
        rgb_tensor(ops, search_rgb, search_width, search_height)?,
    )?;
    let query = ops.normalize(ops.channel_mean(&reference)?)?;
    let fused = ops.fuse(&search, &query)?;
    let object_features = conv_norm(
        ops,
        weights,
        &fused,
        "objectness.0.0",
        "objectness.0.1",
        1,
        1,
        1,
    )?;
    let objectness = conv_bias(ops, weights, &object_features, "objectness.1", 0)?;
    let box_features = conv_norm(ops, weights, &fused, "box.0.0", "box.0.1", 1, 1, 1)?;
    let boxes = conv_bias(ops, weights, &box_features, "box.1", 0)?;
    let scores = ops
        .stream
        .clone_dtoh(&objectness.data)
        .map_err(driver_error)?;
    let distances = ops.stream.clone_dtoh(&boxes.data).map_err(driver_error)?;
    let area = objectness.height * objectness.width;
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
    let mut candidates = Vec::new();
    for (position, logit) in scores.iter().copied().enumerate() {
        let confidence = sigmoid(logit);
        if confidence < minimum_confidence {
            continue;
        }
        let x = (position % objectness.width) as f32 + 0.5;
        let y = (position / objectness.width) as f32 + 0.5;
        let center_x = x / objectness.width as f32;
        let center_y = y / objectness.height as f32;
        candidates.push(Located {
            left: (center_x - sigmoid(distances[position])).clamp(0.0, 1.0),
            top: (center_y - sigmoid(distances[area + position])).clamp(0.0, 1.0),
            right: (center_x + sigmoid(distances[area * 2 + position])).clamp(0.0, 1.0),
            bottom: (center_y + sigmoid(distances[area * 3 + position])).clamp(0.0, 1.0),
            confidence,
        });
    }
    candidates.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut selected: Vec<Located> = Vec::new();
    for candidate in candidates {
        let overlaps = selected
            .iter()
            .any(|existing| intersection_over_union(candidate, *existing) > 0.45);
        if !overlaps {
            selected.push(candidate);
            if selected.len() >= limit {
                break;
            }
        }
    }
    Ok(selected)
}

fn intersection_over_union(left: Located, right: Located) -> f32 {
    let intersection_width = (left.right.min(right.right) - left.left.max(right.left)).max(0.0);
    let intersection_height = (left.bottom.min(right.bottom) - left.top.max(right.top)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let left_area = (left.right - left.left).max(0.0) * (left.bottom - left.top).max(0.0);
    let right_area = (right.right - right.left).max(0.0) * (right.bottom - right.top).max(0.0);
    intersection / (left_area + right_area - intersection).max(1.0e-6)
}

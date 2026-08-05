//! Loader for the compact, external TLN1 neural-network weight container.
//!
//! Training uses PyTorch, but production neither embeds ONNX nor links a
//! framework runtime. The same validated tensor table will feed CPU and GPU
//! execution backends.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

const MAGIC: &[u8; 4] = b"TLN1";
const VERSION: u16 = 1;
const FLOAT16: u8 = 1;
const MAX_TENSORS: usize = 4096;
const MAX_DIMENSIONS: usize = 8;
const MAX_NAME_BYTES: usize = 1024;

#[derive(Debug, PartialEq)]
pub struct TensorRecord {
    pub shape: Vec<usize>,
    pub values: Vec<u16>,
}

#[derive(Debug, PartialEq)]
pub struct ModelWeights {
    pub tensors: BTreeMap<String, TensorRecord>,
    pub parameters: usize,
    pub fingerprint: u64,
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> io::Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| invalid("TLN offset overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("truncated TLN model"))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub fn parse_weights(bytes: &[u8]) -> io::Result<ModelWeights> {
    let fingerprint = bytes.iter().fold(0xcbf29ce484222325u64, |hash, value| {
        (hash ^ u64::from(*value)).wrapping_mul(0x100000001b3)
    });
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != MAGIC {
        return Err(invalid("invalid TLN magic"));
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(invalid(format!("unsupported TLN version {version}")));
    }
    let tensor_count =
        usize::try_from(reader.u32()?).map_err(|_| invalid("tensor count overflow"))?;
    if tensor_count > MAX_TENSORS {
        return Err(invalid("TLN tensor count exceeds the product limit"));
    }

    let mut tensors = BTreeMap::new();
    let mut parameters = 0usize;
    for _ in 0..tensor_count {
        let name_length = usize::from(reader.u16()?);
        let dimensions = usize::from(reader.u8()?);
        let dtype = reader.u8()?;
        if name_length == 0 || name_length > MAX_NAME_BYTES {
            return Err(invalid("invalid TLN tensor name length"));
        }
        if dimensions == 0 || dimensions > MAX_DIMENSIONS || dtype != FLOAT16 {
            return Err(invalid("invalid TLN tensor metadata"));
        }
        let name = std::str::from_utf8(reader.take(name_length)?)
            .map_err(|_| invalid("TLN tensor name is not UTF-8"))?
            .to_owned();

        let mut shape = Vec::with_capacity(dimensions);
        let mut elements = 1usize;
        for _ in 0..dimensions {
            let dimension =
                usize::try_from(reader.u32()?).map_err(|_| invalid("shape overflow"))?;
            if dimension == 0 {
                return Err(invalid("TLN tensor dimensions must be positive"));
            }
            elements = elements
                .checked_mul(dimension)
                .ok_or_else(|| invalid("TLN tensor element count overflow"))?;
            shape.push(dimension);
        }
        let data_length =
            usize::try_from(reader.u64()?).map_err(|_| invalid("data length overflow"))?;
        let expected = elements
            .checked_mul(2)
            .ok_or_else(|| invalid("TLN tensor byte count overflow"))?;
        if data_length != expected {
            return Err(invalid(format!(
                "tensor {name:?} has {data_length} bytes; expected {expected}"
            )));
        }
        let raw = reader.take(data_length)?;
        let values = raw
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        if tensors
            .insert(name.clone(), TensorRecord { shape, values })
            .is_some()
        {
            return Err(invalid(format!("duplicate TLN tensor name {name:?}")));
        }
        parameters = parameters
            .checked_add(elements)
            .ok_or_else(|| invalid("TLN model parameter count overflow"))?;
    }
    if reader.offset != bytes.len() {
        return Err(invalid("TLN model contains trailing data"));
    }
    Ok(ModelWeights {
        tensors,
        parameters,
        fingerprint,
    })
}

pub fn load_weights(path: &Path) -> io::Result<ModelWeights> {
    parse_weights(&fs::read(path)?)
}

type ModelCacheEntry = ((String, u128, u64), Arc<ModelWeights>);
type LocatedTuple = (f32, f32, f32, f32, f32);
static MODEL_CACHE: OnceLock<Mutex<Option<ModelCacheEntry>>> = OnceLock::new();

fn cached_weights(path: &Path) -> io::Result<Arc<ModelWeights>> {
    let resolved = path.canonicalize()?;
    let metadata = fs::metadata(&resolved)?;
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| invalid(error.to_string()))?
        .as_nanos();
    let key = (
        resolved.to_string_lossy().into_owned(),
        modified,
        metadata.len(),
    );
    let cache = MODEL_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache
        .lock()
        .map_err(|_| io::Error::other("TLN model cache is poisoned"))?;
    if let Some((existing_key, model)) = guard.as_ref() {
        if existing_key == &key {
            return Ok(Arc::clone(model));
        }
    }
    let model = Arc::new(load_weights(&resolved)?);
    *guard = Some((key, Arc::clone(&model)));
    Ok(model)
}

#[pyfunction]
fn tinylocate_model_info(path: &str) -> PyResult<(usize, usize, u64)> {
    let metadata = fs::metadata(path).map_err(|error| PyOSError::new_err(error.to_string()))?;
    let model =
        load_weights(Path::new(path)).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((model.tensors.len(), model.parameters, metadata.len()))
}

#[cfg(target_os = "windows")]
#[pyfunction]
#[pyo3(signature = (model_path, reference_rgb, reference_width, reference_height, search_rgb, search_width, search_height))]
#[allow(clippy::too_many_arguments)]
fn tinylocate_infer(
    py: Python<'_>,
    model_path: &str,
    reference_rgb: &[u8],
    reference_width: usize,
    reference_height: usize,
    search_rgb: &[u8],
    search_width: usize,
    search_height: usize,
) -> PyResult<(f32, f32, f32, f32, f32)> {
    let model = cached_weights(Path::new(model_path))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reference = reference_rgb.to_vec();
    let search = search_rgb.to_vec();
    let located = py
        .allow_threads(move || {
            crate::gpu_engine::locate(
                &model,
                &reference,
                reference_width,
                reference_height,
                &search,
                search_width,
                search_height,
            )
        })
        .map_err(|error| PyOSError::new_err(error.to_string()))?;
    Ok((
        located.left,
        located.top,
        located.right,
        located.bottom,
        located.confidence,
    ))
}

#[cfg(target_os = "windows")]
#[pyfunction]
#[pyo3(signature = (model_path, reference_rgb, reference_width, reference_height, search_rgb, search_width, search_height, confidence, limit=128))]
#[allow(clippy::too_many_arguments)]
fn tinylocate_infer_all(
    py: Python<'_>,
    model_path: &str,
    reference_rgb: &[u8],
    reference_width: usize,
    reference_height: usize,
    search_rgb: &[u8],
    search_width: usize,
    search_height: usize,
    confidence: f32,
    limit: usize,
) -> PyResult<Vec<LocatedTuple>> {
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) || limit == 0 || limit > 1024 {
        return Err(PyValueError::new_err(
            "invalid TinyLocate confidence or result limit",
        ));
    }
    let model = cached_weights(Path::new(model_path))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reference = reference_rgb.to_vec();
    let search = search_rgb.to_vec();
    let located = py
        .allow_threads(move || {
            crate::gpu_engine::locate_all(
                &model,
                &reference,
                reference_width,
                reference_height,
                &search,
                search_width,
                search_height,
                confidence,
                limit,
            )
        })
        .map_err(|error| PyOSError::new_err(error.to_string()))?;
    Ok(located
        .into_iter()
        .map(|item| {
            (
                item.left,
                item.top,
                item.right,
                item.bottom,
                item.confidence,
            )
        })
        .collect())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(tinylocate_model_info, module)?)?;
    #[cfg(target_os = "windows")]
    module.add_function(wrap_pyfunction!(tinylocate_infer, module)?)?;
    #[cfg(target_os = "windows")]
    module.add_function(wrap_pyfunction!(tinylocate_infer_all, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_tensor() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&6u16.to_le_bytes());
        bytes.push(2);
        bytes.push(FLOAT16);
        bytes.extend_from_slice(b"weight");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&12u64.to_le_bytes());
        for value in 0u16..6 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn parses_valid_external_weight_container() {
        let model = parse_weights(&one_tensor()).unwrap();
        assert_eq!(model.parameters, 6);
        assert_eq!(model.tensors["weight"].shape, [2, 3]);
        assert_eq!(model.tensors["weight"].values, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn rejects_truncation_and_trailing_data() {
        let mut truncated = one_tensor();
        truncated.pop();
        assert!(parse_weights(&truncated).is_err());

        let mut trailing = one_tensor();
        trailing.push(0);
        assert!(parse_weights(&trailing).is_err());
    }

    #[test]
    fn rejects_wrong_magic_and_tensor_size() {
        let mut wrong_magic = one_tensor();
        wrong_magic[0] = b'X';
        assert!(parse_weights(&wrong_magic).is_err());

        let mut wrong_size = one_tensor();
        let data_length_offset = 4 + 2 + 4 + 2 + 1 + 1 + 6 + 4 + 4;
        wrong_size[data_length_offset..data_length_offset + 8]
            .copy_from_slice(&10u64.to_le_bytes());
        assert!(parse_weights(&wrong_size).is_err());
    }
}

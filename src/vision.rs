//! Allocation-conscious grayscale conversion and template matching primitives.
//!
//! These functions deliberately have no Python or operating-system dependency,
//! which keeps them straightforward to unit-test and benchmark.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemplateMatch {
    pub x: usize,
    pub y: usize,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariantMatch {
    pub matched: TemplateMatch,
    pub variant_index: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug)]
struct SadCandidate {
    x: usize,
    y: usize,
    sad: u64,
    max_sad: u64,
}

#[derive(Clone, Copy)]
struct GrayPlane<'a> {
    pixels: &'a [u8],
    width: usize,
    height: usize,
}

impl SadCandidate {
    fn as_match(self) -> TemplateMatch {
        TemplateMatch {
            x: self.x,
            y: self.y,
            confidence: 1.0 - self.sad as f32 / self.max_sad as f32,
        }
    }
}

/// Convert tightly packed BGRA pixels to luminance using integer BT.601
/// coefficients. The returned buffer contains exactly `width * height` bytes.
pub fn bgra_to_gray(bgra: &[u8], width: usize, height: usize) -> Option<Vec<u8>> {
    let pixels = width.checked_mul(height)?;
    let required = pixels.checked_mul(4)?;
    if bgra.len() < required {
        return None;
    }

    let convert = |pixel: &[u8]| {
        let blue = pixel[0] as u32;
        let green = pixel[1] as u32;
        let red = pixel[2] as u32;
        ((red * 77 + green * 150 + blue * 29) >> 8) as u8
    };
    let mut gray = vec![0; pixels];
    if pixels >= 262_144 {
        gray.par_iter_mut()
            .zip(bgra[..required].par_chunks_exact(4))
            .for_each(|(destination, pixel)| *destination = convert(pixel));
    } else {
        for (destination, pixel) in gray.iter_mut().zip(bgra[..required].chunks_exact(4)) {
            *destination = convert(pixel);
        }
    }
    Some(gray)
}

fn push_top_candidate(top: &mut Vec<SadCandidate>, candidate: SadCandidate, limit: usize) {
    if limit == 0 {
        return;
    }
    if top.len() < limit {
        top.push(candidate);
        return;
    }

    let (worst_index, worst) = top
        .iter()
        .enumerate()
        .max_by_key(|(_, item)| (item.sad, item.y, item.x))
        .expect("non-empty top-candidate collection");
    if (candidate.sad, candidate.y, candidate.x) < (worst.sad, worst.y, worst.x) {
        top[worst_index] = candidate;
    }
}

fn top_sad_matches(
    haystack: GrayPlane<'_>,
    needle: GrayPlane<'_>,
    min_confidence: f32,
    search_rect: Option<(usize, usize, usize, usize)>,
    result_limit: usize,
) -> Vec<SadCandidate> {
    let haystack_len = match haystack.width.checked_mul(haystack.height) {
        Some(value) => value,
        None => return Vec::new(),
    };
    let needle_len = match needle.width.checked_mul(needle.height) {
        Some(value) if value > 0 => value,
        _ => return Vec::new(),
    };
    if haystack.pixels.len() < haystack_len || needle.pixels.len() < needle_len {
        return Vec::new();
    }

    let (start_x, start_y, search_width, search_height) =
        search_rect.unwrap_or((0, 0, haystack.width, haystack.height));
    let search_end_x = match start_x.checked_add(search_width) {
        Some(value) if value <= haystack.width => value,
        _ => return Vec::new(),
    };
    let search_end_y = match start_y.checked_add(search_height) {
        Some(value) if value <= haystack.height => value,
        _ => return Vec::new(),
    };
    if search_width < needle.width || search_height < needle.height {
        return Vec::new();
    }

    let max_sad = match (needle_len as u64).checked_mul(255) {
        Some(value) if value > 0 => value,
        _ => return Vec::new(),
    };
    let allowed_sad =
        ((1.0 - min_confidence.clamp(0.0, 1.0)) as f64 * max_sad as f64).floor() as u64;
    let final_x = search_end_x - needle.width;
    let final_y = search_end_y - needle.height;
    let scan_row = |y| {
        let mut row_results = Vec::with_capacity(result_limit.min(8));
        for x in start_x..=final_x {
            let mut sad = 0u64;
            for needle_y in 0..needle.height {
                let haystack_offset = (y + needle_y) * haystack.width + x;
                let needle_offset = needle_y * needle.width;
                let haystack_row =
                    &haystack.pixels[haystack_offset..haystack_offset + needle.width];
                let needle_row = &needle.pixels[needle_offset..needle_offset + needle.width];
                for (&haystack_value, &needle_value) in haystack_row.iter().zip(needle_row) {
                    sad += haystack_value.abs_diff(needle_value) as u64;
                }
                if sad > allowed_sad {
                    break;
                }
            }

            if sad <= allowed_sad {
                push_top_candidate(
                    &mut row_results,
                    SadCandidate { x, y, sad, max_sad },
                    result_limit,
                );
            }
        }
        row_results
    };

    let positions = (final_x - start_x + 1).saturating_mul(final_y - start_y + 1);
    let mut results: Vec<SadCandidate> = if positions >= 50_000 {
        (start_y..=final_y)
            .into_par_iter()
            .flat_map_iter(|y| scan_row(y).into_iter())
            .collect()
    } else {
        (start_y..=final_y)
            .flat_map(|y| scan_row(y).into_iter())
            .collect()
    };

    results.sort_unstable_by_key(|candidate| (candidate.sad, candidate.y, candidate.x));
    results.truncate(result_limit);
    results
}

fn downsample_phase(
    source: &[u8],
    source_width: usize,
    source_height: usize,
    scale: usize,
    phase_x: usize,
    phase_y: usize,
) -> (Vec<u8>, usize, usize) {
    if scale == 0 || phase_x >= source_width || phase_y >= source_height {
        return (Vec::new(), 0, 0);
    }
    let width = (source_width - phase_x).div_ceil(scale);
    let height = (source_height - phase_y).div_ceil(scale);
    let mut result = Vec::with_capacity(width.saturating_mul(height));
    for y in 0..height {
        let source_y = phase_y + y * scale;
        let row_offset = source_y * source_width;
        for x in 0..width {
            result.push(source[row_offset + phase_x + x * scale]);
        }
    }
    (result, width, height)
}

/// Find one high-confidence match. Large templates use a phase-aware coarse
/// pass followed by full-resolution refinement, avoiding the alignment misses
/// of a single fixed downsampling grid.
pub fn find_best_hierarchical(
    haystack: &[u8],
    haystack_width: usize,
    haystack_height: usize,
    needle: &[u8],
    needle_width: usize,
    needle_height: usize,
    min_confidence: f32,
) -> Option<TemplateMatch> {
    if !min_confidence.is_finite()
        || !(0.0..=1.0).contains(&min_confidence)
        || needle_width == 0
        || needle_height == 0
        || needle_width > haystack_width
        || needle_height > haystack_height
        || haystack.len() < haystack_width.checked_mul(haystack_height)?
        || needle.len() < needle_width.checked_mul(needle_height)?
    {
        return None;
    }

    if needle_width < 32 || needle_height < 32 {
        return top_sad_matches(
            GrayPlane {
                pixels: haystack,
                width: haystack_width,
                height: haystack_height,
            },
            GrayPlane {
                pixels: needle,
                width: needle_width,
                height: needle_height,
            },
            min_confidence,
            None,
            1,
        )
        .into_iter()
        .next()
        .map(SadCandidate::as_match);
    }

    const SCALE: usize = 4;
    const MAX_COARSE_CANDIDATES: usize = 64;
    let (needle_down, needle_down_width, needle_down_height) =
        downsample_phase(needle, needle_width, needle_height, SCALE, 0, 0);
    let coarse_confidence = (min_confidence - 0.12).max(0.45);
    let mut coarse_candidates = Vec::with_capacity(MAX_COARSE_CANDIDATES);

    // Every possible grid phase is searched. Thus an exact match at an
    // arbitrary desktop coordinate is sampled in the same phase as the needle.
    for phase_y in 0..SCALE {
        for phase_x in 0..SCALE {
            let (haystack_down, down_width, down_height) = downsample_phase(
                haystack,
                haystack_width,
                haystack_height,
                SCALE,
                phase_x,
                phase_y,
            );
            for candidate in top_sad_matches(
                GrayPlane {
                    pixels: &haystack_down,
                    width: down_width,
                    height: down_height,
                },
                GrayPlane {
                    pixels: &needle_down,
                    width: needle_down_width,
                    height: needle_down_height,
                },
                coarse_confidence,
                None,
                MAX_COARSE_CANDIDATES,
            ) {
                let mapped = SadCandidate {
                    x: candidate.x * SCALE + phase_x,
                    y: candidate.y * SCALE + phase_y,
                    ..candidate
                };
                push_top_candidate(&mut coarse_candidates, mapped, MAX_COARSE_CANDIDATES);
            }
        }
    }
    coarse_candidates.sort_unstable_by_key(|candidate| (candidate.sad, candidate.y, candidate.x));

    let mut best: Option<SadCandidate> = None;
    for coarse in coarse_candidates {
        let padding = SCALE;
        let start_x = coarse.x.saturating_sub(padding);
        let start_y = coarse.y.saturating_sub(padding);
        let end_x = coarse
            .x
            .saturating_add(needle_width)
            .saturating_add(padding)
            .min(haystack_width);
        let end_y = coarse
            .y
            .saturating_add(needle_height)
            .saturating_add(padding)
            .min(haystack_height);
        if end_x - start_x < needle_width || end_y - start_y < needle_height {
            continue;
        }

        if let Some(candidate) = top_sad_matches(
            GrayPlane {
                pixels: haystack,
                width: haystack_width,
                height: haystack_height,
            },
            GrayPlane {
                pixels: needle,
                width: needle_width,
                height: needle_height,
            },
            min_confidence,
            Some((start_x, start_y, end_x - start_x, end_y - start_y)),
            1,
        )
        .into_iter()
        .next()
        {
            if best
                .map(|old| (candidate.sad, candidate.y, candidate.x) < (old.sad, old.y, old.x))
                .unwrap_or(true)
            {
                best = Some(candidate);
                if candidate.sad == 0 {
                    break;
                }
            }
        }
    }

    best.map(SadCandidate::as_match)
}

/// Search several animation frames and scale variants against one captured
/// image. The capture and grayscale conversion are shared by every variant.
/// Input order is a deterministic tie-breaker, so callers can place the
/// canonical frame before progressively more permissive variants.
pub fn find_best_variant(
    haystack: &[u8],
    haystack_width: usize,
    haystack_height: usize,
    variants: &[(&[u8], usize, usize)],
    min_confidence: f32,
) -> Option<VariantMatch> {
    let mut best: Option<VariantMatch> = None;
    for (variant_index, &(pixels, width, height)) in variants.iter().enumerate() {
        let Some(matched) = find_best_hierarchical(
            haystack,
            haystack_width,
            haystack_height,
            pixels,
            width,
            height,
            min_confidence,
        ) else {
            continue;
        };
        let candidate = VariantMatch {
            matched,
            variant_index,
            width,
            height,
        };
        let replace = best
            .map(|current| {
                candidate.matched.confidence > current.matched.confidence
                    || (candidate.matched.confidence == current.matched.confidence
                        && candidate.variant_index < current.variant_index)
            })
            .unwrap_or(true);
        if replace {
            best = Some(candidate);
        }
        if matched.confidence == 1.0 && variant_index == 0 {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_bgra_to_gray_and_rejects_short_buffers() {
        let gray = bgra_to_gray(&[0, 0, 255, 0, 255, 0, 0, 0], 2, 1).unwrap();
        assert_eq!(gray, vec![76, 28]);
        assert!(bgra_to_gray(&[0, 0, 0], 1, 1).is_none());
    }

    #[test]
    fn small_template_finds_the_best_location() {
        let mut haystack = vec![10u8; 12 * 10];
        let needle = vec![20, 30, 40, 50, 60, 70];
        for y in 0..2 {
            let destination = (4 + y) * 12 + 7;
            haystack[destination..destination + 3].copy_from_slice(&needle[y * 3..y * 3 + 3]);
        }

        let found = find_best_hierarchical(&haystack, 12, 10, &needle, 3, 2, 0.99).unwrap();
        assert_eq!((found.x, found.y), (7, 4));
        assert_eq!(found.confidence, 1.0);
    }

    #[test]
    fn large_template_search_handles_unaligned_coordinates() {
        let width = 101;
        let height = 83;
        let mut haystack = (0..width * height)
            .map(|index| ((index * 37 + index / width * 19) % 251) as u8)
            .collect::<Vec<_>>();
        let needle_width = 36;
        let needle_height = 33;
        let needle = (0..needle_width * needle_height)
            .map(|index| ((index * 73 + index / needle_width * 11 + 7) % 253) as u8)
            .collect::<Vec<_>>();
        let expected_x = 29;
        let expected_y = 27;
        for y in 0..needle_height {
            let destination = (expected_y + y) * width + expected_x;
            haystack[destination..destination + needle_width]
                .copy_from_slice(&needle[y * needle_width..(y + 1) * needle_width]);
        }

        let found = find_best_hierarchical(
            &haystack,
            width,
            height,
            &needle,
            needle_width,
            needle_height,
            0.995,
        )
        .unwrap();
        assert_eq!((found.x, found.y), (expected_x, expected_y));
        assert_eq!(found.confidence, 1.0);
    }

    #[test]
    fn parallel_scan_keeps_deterministic_result_order_and_limit() {
        let width = 300;
        let height = 200;
        let haystack = vec![42; width * height];
        let needle = [42];
        let matches = top_sad_matches(
            GrayPlane {
                pixels: &haystack,
                width,
                height,
            },
            GrayPlane {
                pixels: &needle,
                width: 1,
                height: 1,
            },
            1.0,
            None,
            5,
        );

        assert_eq!(matches.len(), 5);
        assert_eq!(
            matches
                .iter()
                .map(|candidate| (candidate.x, candidate.y))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        );
    }

    #[test]
    fn invalid_dimensions_and_confidence_are_rejected() {
        assert!(find_best_hierarchical(&[0; 4], 2, 2, &[0], 1, 1, f32::NAN).is_none());
        assert!(find_best_hierarchical(&[0; 4], 2, 2, &[], 0, 0, 0.9).is_none());
        assert!(find_best_hierarchical(&[0; 3], 2, 2, &[0], 1, 1, 0.9).is_none());
    }

    #[test]
    fn variant_search_selects_matching_animation_frame() {
        let width = 20;
        let height = 16;
        let mut haystack = vec![10u8; width * height];
        let first = [90u8; 4 * 3];
        let second = (0..12)
            .map(|index| (index * 17 + 20) as u8)
            .collect::<Vec<_>>();
        for y in 0..3 {
            let destination = (7 + y) * width + 9;
            haystack[destination..destination + 4].copy_from_slice(&second[y * 4..(y + 1) * 4]);
        }

        let variants = [(&first[..], 4, 3), (&second[..], 4, 3)];
        let found = find_best_variant(&haystack, width, height, &variants, 0.99).unwrap();
        assert_eq!(found.variant_index, 1);
        assert_eq!((found.matched.x, found.matched.y), (9, 7));
    }
}

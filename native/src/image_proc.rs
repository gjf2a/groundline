use ordered_float::OrderedFloat;
use float_extras::f64::fmod;
use crate::api::ImageData;
use std::cmp::{max, min};

pub type U8ColorTriple = (u8, u8, u8);
pub type F64ColorTriple = (OrderedFloat<f64>, OrderedFloat<f64>, OrderedFloat<f64>);
pub type HueSaturation = (OrderedFloat<f64>, OrderedFloat<f64>);
pub type Uv = (u8, u8);

pub fn color_triple_distance(rgb3a: &U8ColorTriple, rgb3b: &U8ColorTriple) -> f64 {
    (rgb3a.0 as f64 - rgb3b.0 as f64).powf(2.0) + (rgb3a.1 as f64 - rgb3b.1 as f64).powf(2.0) + (rgb3a.2 as f64 - rgb3b.2 as f64).powf(2.0)
}

pub fn color_triple_mean(triples: &Vec<&U8ColorTriple>) -> U8ColorTriple {
    let total = triples.iter().fold((0, 0, 0), |s, t| (s.0 + t.0 as u64, s.1 + t.1 as u64, s.2 + t.2 as u64));
    let count = triples.len() as u64;
    (clamp2u8(total.0 / count), clamp2u8(total.1 / count), clamp2u8(total.2 /count))
}

// Thanks to https://stackoverflow.com/a/35114586/906268
pub fn hs_distance(hsa: &HueSaturation, hsb: &HueSaturation) -> f64 {
    let hdist = OrderedFloat((hsa.0 - hsb.0).abs());
    let hdist = min(hdist, OrderedFloat(360.0) - hdist) / 180.0;
    hdist.powf(2.0) + (hsa.1 - hsb.1).powf(2.0)
}

fn trig_mean<T:Fn(f64)->f64>(hue_sats: &Vec<&HueSaturation>, trig: T) -> f64 {
    hue_sats.iter().map(|(h,_)| trig(h.into_inner().to_radians())).sum::<f64>() / hue_sats.len() as f64
}

// Algorithm from: https://en.wikipedia.org/wiki/Circular_mean
pub fn hs_mean(hue_sats: &Vec<&HueSaturation>) -> HueSaturation {
    let hue_mean = trig_mean(hue_sats, |f| f.sin()).atan2(trig_mean(hue_sats, |f| f.cos()));
    let sat_mean = hue_sats.iter().map(|(_,sat)| sat.into_inner()).sum::<f64>() / hue_sats.len() as f64;
    (OrderedFloat(hue_mean), OrderedFloat(sat_mean))
} 

pub fn simple_yuv_rgb(img: &ImageData) -> Vec<U8ColorTriple> {
    let mut result = vec![];
    generic_yuv_rgba(img, |_,_,rgb| { result.push(rgb)});
    result
}

pub fn hs_values(img: &ImageData) -> Vec<HueSaturation> {
    let mut result = vec![];
    generic_yuv_rgba(img, |_,_,rgb| {
        result.push(rgb2hs(rgb));
    });
    result
}

pub fn uv_values(img: &ImageData) -> Vec<Uv> {
    let mut result = vec![];
    for y in 0..img.height {
        for x in 0..img.width {
            let uv_index = (img.uv_pixel_stride * (x/2) + img.uv_row_stride * (y/2)) as usize;
            result.push((img.us[uv_index], img.vs[uv_index]));
        }
    }
    result
}

pub fn uv_distance(uva: &Uv, uvb: &Uv) -> f64 {
    (uva.0 as f64 - uvb.0 as f64).powf(2.0) + (uva.1 as f64 - uvb.1 as f64).powf(2.0)
}

pub fn uv_mean(uvs: &Vec<&Uv>) -> Uv {
    let total = uvs.iter().fold((0, 0), |s, t| (s.0 + t.0 as u64, s.1 + t.1 as u64));
    let count = uvs.len() as u64;
    (clamp2u8(total.0 / count), clamp2u8(total.1 / count))
}

fn point2index(x: i64, y: i64, width: i64) -> usize {
    ((y * width + x) * 4) as usize
}

/// Translated and adapted from: https://stackoverflow.com/a/57604820/906268
pub fn inner_yuv_rgba(img: &ImageData) -> Vec<u8> {
    let mut result = Vec::new();
    generic_yuv_rgba(img, |_,_,(r, g, b)| {
        result.push(r);
        result.push(g);
        result.push(b);
        result.push(u8::MAX);
    });
    result
}

pub fn generic_yuv_rgba<F: FnMut(i64, i64, (u8, u8, u8))>(img: &ImageData, mut add: F) {
    for y in 0..img.height {
        for x in 0..img.width {
            let uv_index = (img.uv_pixel_stride * (x/2) + img.uv_row_stride * (y/2)) as usize;
            let index = (y * img.width + x) as usize;
            let rgb = yuv2rgb(img.ys[index] as i64, img.us[uv_index] as i64, img.vs[uv_index] as i64);
            add(x, y, rgb);
        }
    }
}

fn yuv2rgb(yp: i64, up: i64, vp: i64) -> (u8, u8, u8) {
    (clamp_u8(yp + vp * 1436 / 1024 - 179), 
     clamp_u8(yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91), 
     clamp_u8(yp + up * 1814 / 1024 - 227))
}

fn scale(b: u8) -> OrderedFloat<f64> {
    OrderedFloat(b as f64 / 255.0)
}

// From https://vocal.com/video/rgb-and-hsvhsihsl-color-space-conversion
fn rgb2hsv(rgb: U8ColorTriple) -> F64ColorTriple {
    let (r, g, b) = rgb;
    let (rf, gf, bf) = (scale(r), scale(g), scale(b));
    let cmax = max(rf, max(gf, bf));
    let cmin = min(rf, min(gf, bf));
    let delta = cmax - cmin;
    let h = OrderedFloat(60.0 * if cmax == rf {
        fmod(((gf - bf) / delta).into_inner(), 6.0)
    } else if cmax == gf {
        ((bf - rf) / delta).into_inner() + 2.0
    } else {
        ((rf - gf) / delta).into_inner() + 4.0
    });
    let s = if cmax == 0.0 {OrderedFloat(0.0)} else {delta / cmax};
    (h, s, cmax)
}

fn rgb2hs(rgb: U8ColorTriple) -> HueSaturation {
    let (h, s, _) = rgb2hsv(rgb);
    (h, s)
}

fn clamp_u8(value: i64) -> u8 {
    min(max(value, 0), u8::MAX as i64) as u8
}

fn clamp2u8(value: u64) -> u8 {
    max(0, min(value, u8::MAX as u64)) as u8
}

pub fn plot(image: &mut Vec<u8>, x: i64, y: i64, width: i64, rgb: U8ColorTriple) {
    let index = point2index(x, y, width);
    if index >= image.len() {
        panic!("Out of bounds at {x}, {y}; width is {width}");
    }
    image[index] = rgb.0;
    image[index + 1] = rgb.1;
    image[index + 2] = rgb.2;
    image[index + 3] = 255;
}

pub fn overlay_points_on(image: &mut Vec<u8>, width: i64, points: &Vec<(i64,i64)>, color: U8ColorTriple) {
    for (x, y) in points.iter() {
        plot(image, *x, *y, width, color);
    }
}
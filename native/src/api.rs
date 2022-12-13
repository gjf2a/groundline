use std::{cmp::{max, min}, collections::BTreeSet, sync::atomic::{Ordering, AtomicBool, AtomicU64}, time::Instant};
use flutter_rust_bridge::{ZeroCopyBuffer};
use correlation_flow::micro_rfft::{COL_DIM, ROW_DIM, MicroFftContext};
use knn::ClusteredKnn;
use std::collections::HashMap;
pub use particle_filter::sonar3bot::{RobotSensorPosition, BOT, MotorData};
use flutter_rust_bridge::support::lazy_static;
use std::sync::{Arc,Mutex};
use kmeans::Kmeans;


type RgbTriple = (u8, u8, u8);

fn rgb_triple_distance(rgb3a: &RgbTriple, rgb3b: &RgbTriple) -> f64 {
    (rgb3a.0 as f64 - rgb3b.0 as f64).powf(2.0) + (rgb3a.1 as f64 - rgb3b.1 as f64).powf(2.0) + (rgb3a.2 as f64 - rgb3b.2 as f64).powf(2.0)
}

fn clamp2u8(value: u64) -> u8 {
    max(0, min(value, u8::MAX as u64)) as u8
}

fn rgb_triple_mean(triples: &Vec<&RgbTriple>) -> RgbTriple {
    let total = triples.iter().fold((0, 0, 0), |s, t| (s.0 + t.0 as u64, s.1 + t.1 as u64, s.2 + t.2 as u64));
    let count = triples.len() as u64;
    (clamp2u8(total.0 / count), clamp2u8(total.1 / count), clamp2u8(total.2 /count))
}

lazy_static! {
    static ref POS: Mutex<RobotSensorPosition> = Mutex::new(RobotSensorPosition::new(BOT));
    static ref COLOR_MEANS: Mutex<Option<Kmeans<RgbTriple, f64, fn (&RgbTriple,&RgbTriple)->f64>>> = Mutex::new(None);
    static ref GROUNDLINE_CLASSIFIER: Mutex<Option<ClusteredKnn<bool, RgbTriple, f64, fn (&RgbTriple,&RgbTriple)->f64, fn (&Vec<&RgbTriple>) -> RgbTriple>>> = Mutex::new(None);
    static ref KMEANS_READY: AtomicBool = AtomicBool::new(false);
    static ref TRAINING_TIME: AtomicU64 = AtomicU64::new(0);
}

pub fn kmeans_ready() -> bool {
    KMEANS_READY.load(Ordering::SeqCst)
}

pub fn training_time() -> i64 {
    TRAINING_TIME.load(Ordering::SeqCst) as i64
}

pub struct ImageData {
    pub ys: Vec<u8>,
    pub us: Vec<u8>,
    pub vs: Vec<u8>,
    pub width: i64,
    pub height: i64,
    pub uv_row_stride: i64,
    pub uv_pixel_stride: i64,
}

pub struct SensorData {
    pub sonar_front: i64,
    pub sonar_left: i64,
    pub sonar_right: i64,
    pub left_count: i64,
    pub right_count: i64,
    pub left_speed: i64,
    pub right_speed: i64
}

impl SensorData {
    fn motor_data(&self) -> MotorData {
        MotorData {left_count: self.left_count, right_count: self.right_count, left_speed: self.left_speed, right_speed: self.right_speed}
    }
}

pub fn intensity_rgba(intensities: Vec<u8>) -> ZeroCopyBuffer<Vec<u8>> {
    let mut result = Vec::new();
    for byte in intensities.iter().copied() {
        for _ in 0..3 {
            result.push(byte);
        }
        result.push(u8::MAX);
    }
    ZeroCopyBuffer(result)
}

pub fn yuv_rgba(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    ZeroCopyBuffer(inner_yuv_rgba(&img))
}

pub fn color_count(img: ImageData) -> i64 {
    let rgba = inner_yuv_rgba(&img);
    let mut distinct_colors = BTreeSet::new();
    for i in (0..rgba.len()).step_by(4) {
        let color = (rgba[i], rgba[i+1], rgba[i+2]);
        distinct_colors.insert(color);
    }
    distinct_colors.len() as i64
}    

const UPPER_SAMPLE_HEIGHT: f64 = 0.25;
const UPPER_SAMPLE_WIDTH: f64 = 0.25;
const LOWER_SAMPLE_WIDTH: f64 = 0.3;
const LOWER_SAMPLE_HEIGHT: f64 = 0.25;

#[derive(Copy, Clone, Debug)]
struct Rect {
    ul_corner: (i64, i64), dimensions: (i64, i64),
}

impl Rect {
    fn place_overlay(&self, image: &mut Vec<u8>, width: i64, color: RgbTriple) {
        for x in self.ul_corner.0..self.ul_corner.0 + self.dimensions.0 {
            plot(image, x, self.ul_corner.1, width, color);
            plot(image, x, self.ul_corner.1 + self.dimensions.1 - 1, width, color);
        }
        for y in self.ul_corner.1..self.ul_corner.1 + self.dimensions.1 {
            plot(image, self.ul_corner.0, y, width, color);
            plot(image, self.ul_corner.0 + self.dimensions.0 - 1, y, width, color);
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct GroundlineOverlay {
    upper: [Rect; 2],
    lower: Rect,
    width: i64, 
}

impl GroundlineOverlay {
    fn new(img: &ImageData) -> Self {
        let upper_max_y = (img.height as f64 * UPPER_SAMPLE_HEIGHT) as i64;
        let upper_left_x_end = (img.width as f64 * UPPER_SAMPLE_WIDTH) as i64;
        let upper_right_x_start = img.width - upper_left_x_end;
        let lower_x_start = (img.width as f64 * (0.5 - LOWER_SAMPLE_WIDTH / 2.0)) as i64;
        let lower_width = (img.width as f64 * LOWER_SAMPLE_WIDTH) as i64;
        let lower_height = (img.height as f64 * LOWER_SAMPLE_HEIGHT) as i64;
        let lower_y_start = (img.height as f64 * (1.0 - LOWER_SAMPLE_HEIGHT)) as i64;
        Self {
            upper: [Rect {ul_corner: (0, 0), dimensions: (upper_left_x_end, upper_max_y)}, Rect {ul_corner: (upper_right_x_start, 0), dimensions: (upper_left_x_end, upper_max_y)}],
            lower: Rect {ul_corner: (lower_x_start, lower_y_start), dimensions: (lower_width, lower_height)},
            width: img.width,
        }
    }

    fn place_overlay(&self, image: &mut Vec<u8>) {
        let white = (255, 255, 255);
        self.upper[0].place_overlay(image, self.width, white);
        self.upper[1].place_overlay(image, self.width, white);
        self.lower.place_overlay(image, self.width, white);    
    }
}

pub fn groundline_sample_overlay(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    let overlay = GroundlineOverlay::new(&img);
    let mut image = inner_yuv_rgba(&img);
    overlay.place_overlay(&mut image);
    ZeroCopyBuffer(image)
}

// 16 clusters yields about 9 frames per second for color filtering on a 176x144 image.
const NUM_COLOR_CLUSTERS: usize = 16;

pub fn start_kmeans_training(img: ImageData) {
    std::thread::spawn(move || {
        let start = Instant::now();
        let image = simple_yuv_rgb(&img);
        let mut color_means = COLOR_MEANS.lock().unwrap();
        *color_means = Some(Kmeans::new(NUM_COLOR_CLUSTERS, &image, Arc::new(rgb_triple_distance), Arc::new(rgb_triple_mean)));
        KMEANS_READY.store(true, Ordering::SeqCst);
        TRAINING_TIME.store(start.elapsed().as_millis() as u64, Ordering::SeqCst);
    });
}

fn cluster_colored(img: ImageData) -> Vec<u8> {
    let image = simple_yuv_rgb(&img);
    COLOR_MEANS.lock().unwrap().as_ref().map_or_else(|| {
        (0..(img.height * img.width * 4)).map(|i| if i % 4 == 0 {u8::MAX} else {0}).collect()
    }, |kmeans| {
        let mut result = vec![];
        for color in image {
            let mean = kmeans.best_matching_mean(&color);
            let bytes: (u8, u8, u8) = mean.into();
            result.push(bytes.0);
            result.push(bytes.1);
            result.push(bytes.2);
            result.push(u8::MAX);
        }
        result
    })
}

pub fn color_clusterer(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    if kmeans_ready() {
        ZeroCopyBuffer(cluster_colored(img))
    } else {
        yuv_rgba(img)
    }
}
 
pub fn groundline_overlay_k_means(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    if kmeans_ready() {
        ZeroCopyBuffer(cluster_colored(img))
    } else {
        groundline_sample_overlay(img)
    }
}

pub fn groundline_filter_k_means(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    if kmeans_ready() {
        ZeroCopyBuffer(cluster_colored(img))
    } else {
        groundline_sample_overlay(img)
    }
}

fn simple_yuv_rgb(img: &ImageData) -> Vec<RgbTriple> {
    let mut result = vec![];
    generic_yuv_rgba(|rgb| { result.push(rgb)}, img);
    result
}

fn point2index(x: i64, y: i64, width: i64) -> usize {
    ((y * width + x) * 4) as usize
}

/// Translated and adapted from: https://stackoverflow.com/a/57604820/906268
fn inner_yuv_rgba(img: &ImageData) -> Vec<u8> {
    let mut result = Vec::new();
    generic_yuv_rgba(|(r, g, b)| {
        result.push(r);
        result.push(g);
        result.push(b);
        result.push(u8::MAX);
    }, img);
    result
}

fn generic_yuv_rgba<F: FnMut((u8, u8, u8))>(mut add: F, img: &ImageData) {
    for y in 0..img.height {
        for x in 0..img.width {
            let uv_index = (img.uv_pixel_stride * (x/2) + img.uv_row_stride * (y/2)) as usize;
            let index = (y * img.width + x) as usize;
            let rgb = yuv2rgb(img.ys[index] as i64, img.us[uv_index] as i64, img.vs[uv_index] as i64);
            add(rgb);
        }
    }
}

fn yuv2rgb(yp: i64, up: i64, vp: i64) -> (u8, u8, u8) {
    (clamp_u8(yp + vp * 1436 / 1024 - 179), 
     clamp_u8(yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91), 
     clamp_u8(yp + up * 1814 / 1024 - 227))
}

fn plot(image: &mut Vec<u8>, x: i64, y: i64, width: i64, color: RgbTriple) {
    let index = point2index(x, y, width);
    if index >= image.len() {
        panic!("Out of bounds at {x}, {y}; width is {width}");
    }
    image[index] = color.0;
    image[index + 1] = color.1;
    image[index + 2] = color.2;
    image[index + 3] = 255;
}

fn overlay_points_on(image: &mut Vec<u8>, width: i64, points: &Vec<(i64,i64)>, color: RgbTriple) {
    for (x, y) in points.iter() {
        plot(image, *x, *y, width, color);
    }
}

fn clamp_u8(value: i64) -> u8 {
    min(max(value, 0), u8::MAX as i64) as u8
}

#[derive(Copy, Clone)]
pub struct CorrelationFlow {
    pub dx: i16, pub dy: i16
}

pub fn get_correlation_flow(prev_ys: Vec<u8>, current_ys: Vec<u8>, width: i64, height: i64) -> CorrelationFlow {
    let mut correlator = MicroFftContext::new();
    let down_prev = downsampled_to(&prev_ys, width, height, COL_DIM as i64, ROW_DIM as i64);
    let down_current = downsampled_to(&current_ys, width, height, COL_DIM as i64, ROW_DIM as i64);
    let (dx, dy) = correlator.measure_translation(&down_prev, &down_current);
    CorrelationFlow {dx, dy}
}

fn downsampled_to(ys: &Vec<u8>, width: i64, height: i64, target_width: i64, target_height: i64) -> Vec<u8> {
    let mut result = Vec::new();
    for y in 0..target_height {
        let old_y = y * height / target_height;
        for x in 0..target_width {
            let old_x = x * width / target_width;
            let i = (old_y * width + old_x) as usize;
            result.push(ys[i]);
        }
    }
    result
}

pub fn reset_position_estimate() {
    let mut pos = POS.lock().unwrap();
    pos.reset();
}

pub fn process_sensor_data(incoming_data: String) -> String {
    let parsed = parse_sensor_data(incoming_data);
    let mut pos = POS.lock().unwrap();
    pos.motor_update(parsed.motor_data());
    let (x, y) = pos.get_pos().position();
    let h = pos.get_pos().heading();
    format!("({x:.2} {y:.2} {h}) {:?} #{}", pos.get_encoder_counts(), pos.num_updates())
}

pub fn parse_sensor_data(incoming_data: String) -> SensorData {
    let parts: HashMap<&str,i64> = incoming_data.split(";").map(|s| {
        let mut ss = s.split(":");
        let key = ss.next().unwrap();
        let value: i64 = ss.next().unwrap().parse().unwrap();
        (key, value)
    }).collect();
    SensorData {
        sonar_front: *parts.get("SF").unwrap(),
        sonar_left: *parts.get("SL").unwrap(),
        sonar_right: *parts.get("SR").unwrap(),
        left_count: *parts.get("LC").unwrap(),
        right_count: *parts.get("RC").unwrap(),
        left_speed: *parts.get("LS").unwrap(),
        right_speed: *parts.get("RS").unwrap(),
    }
}

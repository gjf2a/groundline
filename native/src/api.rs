use std::{collections::BTreeSet, sync::atomic::{Ordering, AtomicBool, AtomicU64}, time::Instant};
use cv::{feature::akaze::Akaze, image::image::DynamicImage};
use flutter_rust_bridge::{ZeroCopyBuffer};
use correlation_flow::micro_rfft::{COL_DIM, ROW_DIM, MicroFftContext};
use image::Rgba;
use knn::ClusteredKnn;
use std::collections::HashMap;
pub use particle_filter::sonar3bot::{RobotSensorPosition, BOT, MotorData};
use flutter_rust_bridge::support::lazy_static;
use std::sync::{Arc,Mutex};
use kmeans::Kmeans;

use crate::{image_proc::{U8ColorTriple, color_triple_distance, color_triple_mean, HueSaturation, hs_values, hs_distance, hs_mean, Uv, uv_values, uv_distance, uv_mean}, groundline::{GroundlineOverlay, HallwayPixel, groundline_pixels}, image_proc::{inner_yuv_rgba, simple_yuv_rgb}, keypoints};

lazy_static! {
    static ref POS: Mutex<RobotSensorPosition> = Mutex::new(RobotSensorPosition::new(BOT));
    static ref RGB_MEANS: Mutex<Option<Kmeans<U8ColorTriple, f64, fn (&U8ColorTriple,&U8ColorTriple)->f64>>> = Mutex::new(None);
    static ref RGB_GROUNDLINE_CLASSIFIER: Mutex<Option<ClusteredKnn<HallwayPixel, U8ColorTriple, f64, fn (&U8ColorTriple,&U8ColorTriple)->f64, fn (&Vec<&U8ColorTriple>) -> U8ColorTriple>>> = Mutex::new(None);
    static ref HS_MEANS: Mutex<Option<Kmeans<HueSaturation, f64, fn (&HueSaturation,&HueSaturation)->f64>>> = Mutex::new(None);
    static ref HS_GROUNDLINE_CLASSIFIER: Mutex<Option<ClusteredKnn<HallwayPixel, HueSaturation, f64, fn (&HueSaturation,&HueSaturation)->f64, fn (&Vec<&HueSaturation>) -> HueSaturation>>> = Mutex::new(None);
    static ref UV_MEANS: Mutex<Option<Kmeans<Uv, f64, fn (&Uv,&Uv)->f64>>> = Mutex::new(None);
    static ref UV_GROUNDLINE_CLASSIFIER: Mutex<Option<ClusteredKnn<HallwayPixel, Uv, f64, fn (&Uv,&Uv)->f64, fn (&Vec<&Uv>) -> Uv>>> = Mutex::new(None);   
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

pub fn groundline_sample_overlay(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    let overlay = GroundlineOverlay::new(&img);
    let mut image = inner_yuv_rgba(&img);
    overlay.place_overlay(&mut image);
    ZeroCopyBuffer(image)
}

// 16 clusters yields about 9 frames per second for color filtering on a 176x144 image.
const NUM_COLOR_CLUSTERS: usize = 16;
const NUM_KNN_REFS: usize = 3;

pub fn start_rgb_kmeans_training(img: ImageData) {
    std::thread::spawn(move || {
        let start = Instant::now();
        let image = simple_yuv_rgb(&img);
        let mut color_means = RGB_MEANS.lock().unwrap();
        *color_means = Some(Kmeans::new(NUM_COLOR_CLUSTERS, &image, Arc::new(color_triple_distance), Arc::new(color_triple_mean)));
        let mut groundline_classifier = RGB_GROUNDLINE_CLASSIFIER.lock().unwrap();
        *groundline_classifier = Some(ClusteredKnn::new(NUM_KNN_REFS, NUM_COLOR_CLUSTERS, Arc::new(color_triple_distance), Arc::new(color_triple_mean)));
        let overlay = GroundlineOverlay::new(&img);
        let training_examples = overlay.make_color_training_from(&image, img.width);
        groundline_classifier.as_mut().unwrap().train_from_clusters(color_means.as_ref().unwrap(), &training_examples);
        KMEANS_READY.store(true, Ordering::SeqCst);
        TRAINING_TIME.store(start.elapsed().as_millis() as u64, Ordering::SeqCst);
    });
}

fn cluster_colored(img: ImageData) -> Vec<u8> {
    let image = simple_yuv_rgb(&img);
    RGB_MEANS.lock().unwrap().as_ref().map_or_else(|| {
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

fn ground_colored(img: ImageData) -> Vec<u8> {
    let image = simple_yuv_rgb(&img);
    RGB_GROUNDLINE_CLASSIFIER.lock().unwrap().as_ref().map_or_else(|| {
        (0..(img.height * img.width * 4)).map(|i| if i % 4 == 0 {u8::MAX} else {0}).collect()
    }, |knn| groundline_pixels(&image, knn))
}

pub fn start_hs_kmeans_training(img: ImageData) {
    std::thread::spawn(move || {
        let start = Instant::now();
        let image = hs_values(&img);
        let mut color_means = HS_MEANS.lock().unwrap();
        *color_means = Some(Kmeans::new(NUM_COLOR_CLUSTERS, &image, Arc::new(hs_distance), Arc::new(hs_mean)));
        let mut groundline_classifier = HS_GROUNDLINE_CLASSIFIER.lock().unwrap();
        *groundline_classifier = Some(ClusteredKnn::new(NUM_KNN_REFS, NUM_COLOR_CLUSTERS, Arc::new(hs_distance), Arc::new(hs_mean)));
        let overlay = GroundlineOverlay::new(&img);
        let training_examples = overlay.make_color_training_from(&image, img.width);
        groundline_classifier.as_mut().unwrap().train_from_clusters(color_means.as_ref().unwrap(), &training_examples);
        KMEANS_READY.store(true, Ordering::SeqCst);
        TRAINING_TIME.store(start.elapsed().as_millis() as u64, Ordering::SeqCst);
    });
}

pub fn start_uv_kmeans_training(img: ImageData) {
    std::thread::spawn(move || {
        let start = Instant::now();
        let image = uv_values(&img);
        let mut color_means = UV_MEANS.lock().unwrap();
        *color_means = Some(Kmeans::new(NUM_COLOR_CLUSTERS, &image, Arc::new(uv_distance), Arc::new(uv_mean)));
        let mut groundline_classifier = UV_GROUNDLINE_CLASSIFIER.lock().unwrap();
        *groundline_classifier = Some(ClusteredKnn::new(NUM_KNN_REFS, NUM_COLOR_CLUSTERS, Arc::new(uv_distance), Arc::new(uv_mean)));
        let overlay = GroundlineOverlay::new(&img);
        let training_examples = overlay.make_color_training_from(&image, img.width);
        groundline_classifier.as_mut().unwrap().train_from_clusters(color_means.as_ref().unwrap(), &training_examples);
        KMEANS_READY.store(true, Ordering::SeqCst);
        TRAINING_TIME.store(start.elapsed().as_millis() as u64, Ordering::SeqCst);
    });
}

fn white_screen_of_doom(img: &ImageData) -> Vec<u8> {
    (0..(img.height * img.width * 4)).map(|i| if i % 4 == 0 {u8::MAX} else {0}).collect()
}

fn hs_ground_colored(img: ImageData) -> Vec<u8> {
    let image = hs_values(&img);
    HS_GROUNDLINE_CLASSIFIER.lock().unwrap().as_ref().map_or_else(|| white_screen_of_doom(&img), 
    |knn| groundline_pixels(&image, knn))
}

fn uv_ground_colored(img: ImageData) -> Vec<u8> {
    let image = uv_values(&img);
    UV_GROUNDLINE_CLASSIFIER.lock().unwrap().as_ref().map_or_else(|| white_screen_of_doom(&img), 
    |knn| groundline_pixels(&image, knn))
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
        ZeroCopyBuffer(ground_colored(img))
    } else {
        groundline_sample_overlay(img)
    }
}

pub fn hs_groundline_filter_k_means(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    if kmeans_ready() {
        ZeroCopyBuffer(hs_ground_colored(img))
    } else {
        groundline_sample_overlay(img)
    }
}

pub fn uv_groundline_filter_k_means(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    if kmeans_ready() {
        ZeroCopyBuffer(uv_ground_colored(img))
    } else {
        groundline_sample_overlay(img)
    }
}

pub fn akaze_view(img: ImageData) -> ZeroCopyBuffer<Vec<u8>> {
    let rgba = keypoints::convert(&img);
    let wrapped = DynamicImage::ImageRgba8(rgba);
    let akaze = Akaze::dense();
    let (keypoints, _) = akaze.extract(&wrapped);
    if let DynamicImage::ImageRgba8(mut unwrapped) = wrapped {
        for kp in keypoints {
            let (x, y) = kp.point;
            unwrapped.put_pixel(x as u32, y as u32, Rgba([255, 0, 0, 255]));
        }
        ZeroCopyBuffer(unwrapped.into_vec())
    } else {
        panic!("This shouldn't happen");
    }
    
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

use knn::ClusteredKnn;
use supervised_learning::Classifier;

use crate::{image_proc::U8ColorTriple, api::ImageData, image_proc::plot};

pub trait ColorType: Copy + Clone + Eq + PartialEq {}
impl <T: Copy + Clone + Eq + PartialEq> ColorType for T {}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Ord, PartialOrd)]
pub enum HallwayPixel {
    Ground, Elevated
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
    fn place_overlay(&self, image: &mut Vec<u8>, width: i64, color: U8ColorTriple) {
        for x in self.ul_corner.0..self.ul_corner.0 + self.dimensions.0 {
            plot(image, x, self.ul_corner.1, width, color);
            plot(image, x, self.ul_corner.1 + self.dimensions.1 - 1, width, color);
        }
        for y in self.ul_corner.1..self.ul_corner.1 + self.dimensions.1 {
            plot(image, self.ul_corner.0, y, width, color);
            plot(image, self.ul_corner.0 + self.dimensions.0 - 1, y, width, color);
        }
    }

    fn indices_in(&self, width: i64) -> Vec<usize> {
        let mut result = vec![];
        for x in self.ul_corner.0..self.ul_corner.0 + self.dimensions.0 {
            for y in self.ul_corner.1..self.ul_corner.1 + self.dimensions.1 {
                result.push((y * width + x) as usize);
            }
        }
        result
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GroundlineOverlay {
    upper: [Rect; 2],
    lower: Rect,
    width: i64, 
}

impl GroundlineOverlay {
    pub fn new(img: &ImageData) -> Self {
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

    pub fn place_overlay(&self, image: &mut Vec<u8>) {
        let white = (255, 255, 255);
        self.upper[0].place_overlay(image, self.width, white);
        self.upper[1].place_overlay(image, self.width, white);
        self.lower.place_overlay(image, self.width, white);    
    }

    fn visit_upper_pixels<C: ColorType, V, F:FnMut(&mut V, C)>(&self, img: &Vec<C>, width: i64, mut visit: F, storage: &mut V) {
        for rect in self.upper.iter() {
            for i in rect.indices_in(width) {
                visit(storage, img[i]);
            }
        }
    }

    fn visit_lower_pixels<C: ColorType, V, F:FnMut(&mut V, C)>(&self, img: &Vec<C>, width: i64, mut visit: F, storage: &mut V) {
        for i in self.lower.indices_in(width) {
            visit(storage, img[i]);
        }
    }

    pub fn make_color_training_from<C: ColorType>(&self, img: &Vec<C>, width: i64) -> Vec<(HallwayPixel, C)> {
        let mut result = vec![];
        self.visit_upper_pixels(img, width, |v, p| v.push((HallwayPixel::Elevated, p)), &mut result);
        self.visit_lower_pixels(img, width, |v, p| v.push((HallwayPixel::Ground, p)), &mut result);
        result
    }
}

pub fn groundline_pixels<C:ColorType>(image: &Vec<C>, knn: &ClusteredKnn<HallwayPixel, C, f64, fn (&C,&C)->f64, fn (&Vec<&C>) -> C>) -> Vec<u8> {
    let mut result = vec![];
    for color in image {
        let bytes = match knn.classify(&color) {
            HallwayPixel::Ground => (u8::MAX, 0, 0),
            HallwayPixel::Elevated => (0, 0, u8::MAX),
        };
        
        result.push(bytes.0);
        result.push(bytes.1);
        result.push(bytes.2);
        result.push(u8::MAX);
    }
    result
}
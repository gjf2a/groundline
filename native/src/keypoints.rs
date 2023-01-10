use image::{RgbaImage, Rgba};

use crate::{api::ImageData, image_proc::generic_yuv_rgba};

pub fn convert(img: &ImageData) -> RgbaImage {
    let mut result = RgbaImage::new(img.width as u32, img.height as u32);
    generic_yuv_rgba(&img, |x, y, (r, g, b)| {
        result.put_pixel(x as u32, y as u32, Rgba([r, g, b, 100]));
    });
    result
}


#![allow(
    non_camel_case_types,
    unused,
    clippy::redundant_closure,
    clippy::useless_conversion,
    clippy::unit_arg,
    clippy::double_parens,
    non_snake_case,
    clippy::too_many_arguments
)]
// AUTO GENERATED FILE, DO NOT EDIT.
// Generated by `flutter_rust_bridge`@ 1.49.0.

use crate::api::*;
use core::panic::UnwindSafe;
use flutter_rust_bridge::*;

// Section: imports

// Section: wire functions

fn wire_kmeans_ready_impl(port_: MessagePort) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "kmeans_ready",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || move |task_callback| Ok(kmeans_ready()),
    )
}
fn wire_training_time_impl(port_: MessagePort) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "training_time",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || move |task_callback| Ok(training_time()),
    )
}
fn wire_intensity_rgba_impl(port_: MessagePort, intensities: impl Wire2Api<Vec<u8>> + UnwindSafe) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "intensity_rgba",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_intensities = intensities.wire2api();
            move |task_callback| Ok(intensity_rgba(api_intensities))
        },
    )
}
fn wire_yuv_rgba_impl(port_: MessagePort, img: impl Wire2Api<ImageData> + UnwindSafe) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "yuv_rgba",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_img = img.wire2api();
            move |task_callback| Ok(yuv_rgba(api_img))
        },
    )
}
fn wire_color_count_impl(port_: MessagePort, img: impl Wire2Api<ImageData> + UnwindSafe) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "color_count",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_img = img.wire2api();
            move |task_callback| Ok(color_count(api_img))
        },
    )
}
fn wire_groundline_sample_overlay_impl(
    port_: MessagePort,
    img: impl Wire2Api<ImageData> + UnwindSafe,
) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "groundline_sample_overlay",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_img = img.wire2api();
            move |task_callback| Ok(groundline_sample_overlay(api_img))
        },
    )
}
fn wire_start_kmeans_training_impl(port_: MessagePort, img: impl Wire2Api<ImageData> + UnwindSafe) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "start_kmeans_training",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_img = img.wire2api();
            move |task_callback| Ok(start_kmeans_training(api_img))
        },
    )
}
fn wire_color_clusterer_impl(port_: MessagePort, img: impl Wire2Api<ImageData> + UnwindSafe) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "color_clusterer",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_img = img.wire2api();
            move |task_callback| Ok(color_clusterer(api_img))
        },
    )
}
fn wire_groundline_k_means_impl(port_: MessagePort, img: impl Wire2Api<ImageData> + UnwindSafe) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "groundline_k_means",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_img = img.wire2api();
            move |task_callback| Ok(groundline_k_means(api_img))
        },
    )
}
fn wire_get_correlation_flow_impl(
    port_: MessagePort,
    prev_ys: impl Wire2Api<Vec<u8>> + UnwindSafe,
    current_ys: impl Wire2Api<Vec<u8>> + UnwindSafe,
    width: impl Wire2Api<i64> + UnwindSafe,
    height: impl Wire2Api<i64> + UnwindSafe,
) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "get_correlation_flow",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_prev_ys = prev_ys.wire2api();
            let api_current_ys = current_ys.wire2api();
            let api_width = width.wire2api();
            let api_height = height.wire2api();
            move |task_callback| {
                Ok(get_correlation_flow(
                    api_prev_ys,
                    api_current_ys,
                    api_width,
                    api_height,
                ))
            }
        },
    )
}
fn wire_reset_position_estimate_impl(port_: MessagePort) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "reset_position_estimate",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || move |task_callback| Ok(reset_position_estimate()),
    )
}
fn wire_process_sensor_data_impl(
    port_: MessagePort,
    incoming_data: impl Wire2Api<String> + UnwindSafe,
) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "process_sensor_data",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_incoming_data = incoming_data.wire2api();
            move |task_callback| Ok(process_sensor_data(api_incoming_data))
        },
    )
}
fn wire_parse_sensor_data_impl(
    port_: MessagePort,
    incoming_data: impl Wire2Api<String> + UnwindSafe,
) {
    FLUTTER_RUST_BRIDGE_HANDLER.wrap(
        WrapInfo {
            debug_name: "parse_sensor_data",
            port: Some(port_),
            mode: FfiCallMode::Normal,
        },
        move || {
            let api_incoming_data = incoming_data.wire2api();
            move |task_callback| Ok(parse_sensor_data(api_incoming_data))
        },
    )
}
// Section: wrapper structs

// Section: static checks

// Section: allocate functions

// Section: impl Wire2Api

pub trait Wire2Api<T> {
    fn wire2api(self) -> T;
}

impl<T, S> Wire2Api<Option<T>> for *mut S
where
    *mut S: Wire2Api<T>,
{
    fn wire2api(self) -> Option<T> {
        (!self.is_null()).then(|| self.wire2api())
    }
}

impl Wire2Api<i64> for i64 {
    fn wire2api(self) -> i64 {
        self
    }
}

impl Wire2Api<u8> for u8 {
    fn wire2api(self) -> u8 {
        self
    }
}

// Section: impl IntoDart

impl support::IntoDart for CorrelationFlow {
    fn into_dart(self) -> support::DartAbi {
        vec![self.dx.into_dart(), self.dy.into_dart()].into_dart()
    }
}
impl support::IntoDartExceptPrimitive for CorrelationFlow {}

impl support::IntoDart for SensorData {
    fn into_dart(self) -> support::DartAbi {
        vec![
            self.sonar_front.into_dart(),
            self.sonar_left.into_dart(),
            self.sonar_right.into_dart(),
            self.left_count.into_dart(),
            self.right_count.into_dart(),
            self.left_speed.into_dart(),
            self.right_speed.into_dart(),
        ]
        .into_dart()
    }
}
impl support::IntoDartExceptPrimitive for SensorData {}

// Section: executor

support::lazy_static! {
    pub static ref FLUTTER_RUST_BRIDGE_HANDLER: support::DefaultHandler = Default::default();
}

#[cfg(not(target_family = "wasm"))]
#[path = "bridge_generated.io.rs"]
mod io;
#[cfg(not(target_family = "wasm"))]
pub use io::*;
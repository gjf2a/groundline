// AUTO GENERATED FILE, DO NOT EDIT.
// Generated by `flutter_rust_bridge`@ 1.54.1.
// ignore_for_file: non_constant_identifier_names, unused_element, duplicate_ignore, directives_ordering, curly_braces_in_flow_control_structures, unnecessary_lambdas, slash_for_doc_comments, prefer_const_literals_to_create_immutables, implicit_dynamic_list_literal, duplicate_import, unused_import, prefer_single_quotes, prefer_const_constructors, use_super_parameters, always_use_package_imports, annotate_overrides, invalid_use_of_protected_member, constant_identifier_names, invalid_use_of_internal_member

import "bridge_definitions.dart";
import 'dart:convert';
import 'dart:async';
import 'package:flutter_rust_bridge/flutter_rust_bridge.dart';

import 'package:meta/meta.dart';
import 'dart:convert';
import 'dart:async';
import 'package:flutter_rust_bridge/flutter_rust_bridge.dart';
import 'package:meta/meta.dart';
import 'dart:ffi' as ffi;

class NativeImpl implements Native {
  final NativePlatform _platform;
  factory NativeImpl(ExternalLibrary dylib) =>
      NativeImpl.raw(NativePlatform(dylib));

  /// Only valid on web/WASM platforms.
  factory NativeImpl.wasm(FutureOr<WasmModule> module) =>
      NativeImpl(module as ExternalLibrary);
  NativeImpl.raw(this._platform);
  Future<bool> kmeansReady({dynamic hint}) {
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_kmeans_ready(port_),
      parseSuccessData: _wire2api_bool,
      constMeta: kKmeansReadyConstMeta,
      argValues: [],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kKmeansReadyConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "kmeans_ready",
        argNames: [],
      );

  Future<int> trainingTime({dynamic hint}) {
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_training_time(port_),
      parseSuccessData: _wire2api_i64,
      constMeta: kTrainingTimeConstMeta,
      argValues: [],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kTrainingTimeConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "training_time",
        argNames: [],
      );

  Future<Uint8List> intensityRgba(
      {required Uint8List intensities, dynamic hint}) {
    var arg0 = _platform.api2wire_uint_8_list(intensities);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_intensity_rgba(port_, arg0),
      parseSuccessData: _wire2api_ZeroCopyBuffer_Uint8List,
      constMeta: kIntensityRgbaConstMeta,
      argValues: [intensities],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kIntensityRgbaConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "intensity_rgba",
        argNames: ["intensities"],
      );

  Future<Uint8List> yuvRgba({required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_yuv_rgba(port_, arg0),
      parseSuccessData: _wire2api_ZeroCopyBuffer_Uint8List,
      constMeta: kYuvRgbaConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kYuvRgbaConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "yuv_rgba",
        argNames: ["img"],
      );

  Future<int> colorCount({required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_color_count(port_, arg0),
      parseSuccessData: _wire2api_i64,
      constMeta: kColorCountConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kColorCountConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "color_count",
        argNames: ["img"],
      );

  Future<Uint8List> groundlineSampleOverlay(
      {required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) =>
          _platform.inner.wire_groundline_sample_overlay(port_, arg0),
      parseSuccessData: _wire2api_ZeroCopyBuffer_Uint8List,
      constMeta: kGroundlineSampleOverlayConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kGroundlineSampleOverlayConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "groundline_sample_overlay",
        argNames: ["img"],
      );

  Future<void> startRgbKmeansTraining({required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) =>
          _platform.inner.wire_start_rgb_kmeans_training(port_, arg0),
      parseSuccessData: _wire2api_unit,
      constMeta: kStartRgbKmeansTrainingConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kStartRgbKmeansTrainingConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "start_rgb_kmeans_training",
        argNames: ["img"],
      );

  Future<void> startHsKmeansTraining({required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) =>
          _platform.inner.wire_start_hs_kmeans_training(port_, arg0),
      parseSuccessData: _wire2api_unit,
      constMeta: kStartHsKmeansTrainingConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kStartHsKmeansTrainingConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "start_hs_kmeans_training",
        argNames: ["img"],
      );

  Future<Uint8List> colorClusterer({required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_color_clusterer(port_, arg0),
      parseSuccessData: _wire2api_ZeroCopyBuffer_Uint8List,
      constMeta: kColorClustererConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kColorClustererConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "color_clusterer",
        argNames: ["img"],
      );

  Future<Uint8List> groundlineOverlayKMeans(
      {required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) =>
          _platform.inner.wire_groundline_overlay_k_means(port_, arg0),
      parseSuccessData: _wire2api_ZeroCopyBuffer_Uint8List,
      constMeta: kGroundlineOverlayKMeansConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kGroundlineOverlayKMeansConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "groundline_overlay_k_means",
        argNames: ["img"],
      );

  Future<Uint8List> groundlineFilterKMeans(
      {required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) =>
          _platform.inner.wire_groundline_filter_k_means(port_, arg0),
      parseSuccessData: _wire2api_ZeroCopyBuffer_Uint8List,
      constMeta: kGroundlineFilterKMeansConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kGroundlineFilterKMeansConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "groundline_filter_k_means",
        argNames: ["img"],
      );

  Future<Uint8List> hsGroundlineFilterKMeans(
      {required ImageData img, dynamic hint}) {
    var arg0 = _platform.api2wire_box_autoadd_image_data(img);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) =>
          _platform.inner.wire_hs_groundline_filter_k_means(port_, arg0),
      parseSuccessData: _wire2api_ZeroCopyBuffer_Uint8List,
      constMeta: kHsGroundlineFilterKMeansConstMeta,
      argValues: [img],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kHsGroundlineFilterKMeansConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "hs_groundline_filter_k_means",
        argNames: ["img"],
      );

  Future<CorrelationFlow> getCorrelationFlow(
      {required Uint8List prevYs,
      required Uint8List currentYs,
      required int width,
      required int height,
      dynamic hint}) {
    var arg0 = _platform.api2wire_uint_8_list(prevYs);
    var arg1 = _platform.api2wire_uint_8_list(currentYs);
    var arg2 = _platform.api2wire_i64(width);
    var arg3 = _platform.api2wire_i64(height);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner
          .wire_get_correlation_flow(port_, arg0, arg1, arg2, arg3),
      parseSuccessData: _wire2api_correlation_flow,
      constMeta: kGetCorrelationFlowConstMeta,
      argValues: [prevYs, currentYs, width, height],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kGetCorrelationFlowConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "get_correlation_flow",
        argNames: ["prevYs", "currentYs", "width", "height"],
      );

  Future<void> resetPositionEstimate({dynamic hint}) {
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_reset_position_estimate(port_),
      parseSuccessData: _wire2api_unit,
      constMeta: kResetPositionEstimateConstMeta,
      argValues: [],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kResetPositionEstimateConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "reset_position_estimate",
        argNames: [],
      );

  Future<String> processSensorData(
      {required String incomingData, dynamic hint}) {
    var arg0 = _platform.api2wire_String(incomingData);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_process_sensor_data(port_, arg0),
      parseSuccessData: _wire2api_String,
      constMeta: kProcessSensorDataConstMeta,
      argValues: [incomingData],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kProcessSensorDataConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "process_sensor_data",
        argNames: ["incomingData"],
      );

  Future<SensorData> parseSensorData(
      {required String incomingData, dynamic hint}) {
    var arg0 = _platform.api2wire_String(incomingData);
    return _platform.executeNormal(FlutterRustBridgeTask(
      callFfi: (port_) => _platform.inner.wire_parse_sensor_data(port_, arg0),
      parseSuccessData: _wire2api_sensor_data,
      constMeta: kParseSensorDataConstMeta,
      argValues: [incomingData],
      hint: hint,
    ));
  }

  FlutterRustBridgeTaskConstMeta get kParseSensorDataConstMeta =>
      const FlutterRustBridgeTaskConstMeta(
        debugName: "parse_sensor_data",
        argNames: ["incomingData"],
      );

  void dispose() {
    _platform.dispose();
  }
// Section: wire2api

  String _wire2api_String(dynamic raw) {
    return raw as String;
  }

  Uint8List _wire2api_ZeroCopyBuffer_Uint8List(dynamic raw) {
    return raw as Uint8List;
  }

  bool _wire2api_bool(dynamic raw) {
    return raw as bool;
  }

  CorrelationFlow _wire2api_correlation_flow(dynamic raw) {
    final arr = raw as List<dynamic>;
    if (arr.length != 2)
      throw Exception('unexpected arr length: expect 2 but see ${arr.length}');
    return CorrelationFlow(
      dx: _wire2api_i16(arr[0]),
      dy: _wire2api_i16(arr[1]),
    );
  }

  int _wire2api_i16(dynamic raw) {
    return raw as int;
  }

  int _wire2api_i64(dynamic raw) {
    return castInt(raw);
  }

  SensorData _wire2api_sensor_data(dynamic raw) {
    final arr = raw as List<dynamic>;
    if (arr.length != 7)
      throw Exception('unexpected arr length: expect 7 but see ${arr.length}');
    return SensorData(
      sonarFront: _wire2api_i64(arr[0]),
      sonarLeft: _wire2api_i64(arr[1]),
      sonarRight: _wire2api_i64(arr[2]),
      leftCount: _wire2api_i64(arr[3]),
      rightCount: _wire2api_i64(arr[4]),
      leftSpeed: _wire2api_i64(arr[5]),
      rightSpeed: _wire2api_i64(arr[6]),
    );
  }

  int _wire2api_u8(dynamic raw) {
    return raw as int;
  }

  Uint8List _wire2api_uint_8_list(dynamic raw) {
    return raw as Uint8List;
  }

  void _wire2api_unit(dynamic raw) {
    return;
  }
}

// Section: api2wire

@protected
int api2wire_u8(int raw) {
  return raw;
}

// Section: finalizer

class NativePlatform extends FlutterRustBridgeBase<NativeWire> {
  NativePlatform(ffi.DynamicLibrary dylib) : super(NativeWire(dylib));

// Section: api2wire

  @protected
  ffi.Pointer<wire_uint_8_list> api2wire_String(String raw) {
    return api2wire_uint_8_list(utf8.encoder.convert(raw));
  }

  @protected
  ffi.Pointer<wire_ImageData> api2wire_box_autoadd_image_data(ImageData raw) {
    final ptr = inner.new_box_autoadd_image_data_0();
    _api_fill_to_wire_image_data(raw, ptr.ref);
    return ptr;
  }

  @protected
  int api2wire_i64(int raw) {
    return raw;
  }

  @protected
  ffi.Pointer<wire_uint_8_list> api2wire_uint_8_list(Uint8List raw) {
    final ans = inner.new_uint_8_list_0(raw.length);
    ans.ref.ptr.asTypedList(raw.length).setAll(0, raw);
    return ans;
  }
// Section: finalizer

// Section: api_fill_to_wire

  void _api_fill_to_wire_box_autoadd_image_data(
      ImageData apiObj, ffi.Pointer<wire_ImageData> wireObj) {
    _api_fill_to_wire_image_data(apiObj, wireObj.ref);
  }

  void _api_fill_to_wire_image_data(ImageData apiObj, wire_ImageData wireObj) {
    wireObj.ys = api2wire_uint_8_list(apiObj.ys);
    wireObj.us = api2wire_uint_8_list(apiObj.us);
    wireObj.vs = api2wire_uint_8_list(apiObj.vs);
    wireObj.width = api2wire_i64(apiObj.width);
    wireObj.height = api2wire_i64(apiObj.height);
    wireObj.uv_row_stride = api2wire_i64(apiObj.uvRowStride);
    wireObj.uv_pixel_stride = api2wire_i64(apiObj.uvPixelStride);
  }
}

// ignore_for_file: camel_case_types, non_constant_identifier_names, avoid_positional_boolean_parameters, annotate_overrides, constant_identifier_names

// AUTO GENERATED FILE, DO NOT EDIT.
//
// Generated by `package:ffigen`.

/// generated by flutter_rust_bridge
class NativeWire implements FlutterRustBridgeWireBase {
  @internal
  late final dartApi = DartApiDl(init_frb_dart_api_dl);

  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  NativeWire(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  NativeWire.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  void store_dart_post_cobject(
    DartPostCObjectFnType ptr,
  ) {
    return _store_dart_post_cobject(
      ptr,
    );
  }

  late final _store_dart_post_cobjectPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(DartPostCObjectFnType)>>(
          'store_dart_post_cobject');
  late final _store_dart_post_cobject = _store_dart_post_cobjectPtr
      .asFunction<void Function(DartPostCObjectFnType)>();

  Object get_dart_object(
    int ptr,
  ) {
    return _get_dart_object(
      ptr,
    );
  }

  late final _get_dart_objectPtr =
      _lookup<ffi.NativeFunction<ffi.Handle Function(uintptr_t)>>(
          'get_dart_object');
  late final _get_dart_object =
      _get_dart_objectPtr.asFunction<Object Function(int)>();

  void drop_dart_object(
    int ptr,
  ) {
    return _drop_dart_object(
      ptr,
    );
  }

  late final _drop_dart_objectPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(uintptr_t)>>(
          'drop_dart_object');
  late final _drop_dart_object =
      _drop_dart_objectPtr.asFunction<void Function(int)>();

  int new_dart_opaque(
    Object handle,
  ) {
    return _new_dart_opaque(
      handle,
    );
  }

  late final _new_dart_opaquePtr =
      _lookup<ffi.NativeFunction<uintptr_t Function(ffi.Handle)>>(
          'new_dart_opaque');
  late final _new_dart_opaque =
      _new_dart_opaquePtr.asFunction<int Function(Object)>();

  int init_frb_dart_api_dl(
    ffi.Pointer<ffi.Void> obj,
  ) {
    return _init_frb_dart_api_dl(
      obj,
    );
  }

  late final _init_frb_dart_api_dlPtr =
      _lookup<ffi.NativeFunction<ffi.IntPtr Function(ffi.Pointer<ffi.Void>)>>(
          'init_frb_dart_api_dl');
  late final _init_frb_dart_api_dl = _init_frb_dart_api_dlPtr
      .asFunction<int Function(ffi.Pointer<ffi.Void>)>();

  void wire_kmeans_ready(
    int port_,
  ) {
    return _wire_kmeans_ready(
      port_,
    );
  }

  late final _wire_kmeans_readyPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(ffi.Int64)>>(
          'wire_kmeans_ready');
  late final _wire_kmeans_ready =
      _wire_kmeans_readyPtr.asFunction<void Function(int)>();

  void wire_training_time(
    int port_,
  ) {
    return _wire_training_time(
      port_,
    );
  }

  late final _wire_training_timePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(ffi.Int64)>>(
          'wire_training_time');
  late final _wire_training_time =
      _wire_training_timePtr.asFunction<void Function(int)>();

  void wire_intensity_rgba(
    int port_,
    ffi.Pointer<wire_uint_8_list> intensities,
  ) {
    return _wire_intensity_rgba(
      port_,
      intensities,
    );
  }

  late final _wire_intensity_rgbaPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_uint_8_list>)>>('wire_intensity_rgba');
  late final _wire_intensity_rgba = _wire_intensity_rgbaPtr
      .asFunction<void Function(int, ffi.Pointer<wire_uint_8_list>)>();

  void wire_yuv_rgba(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_yuv_rgba(
      port_,
      img,
    );
  }

  late final _wire_yuv_rgbaPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(
              ffi.Int64, ffi.Pointer<wire_ImageData>)>>('wire_yuv_rgba');
  late final _wire_yuv_rgba = _wire_yuv_rgbaPtr
      .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_color_count(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_color_count(
      port_,
      img,
    );
  }

  late final _wire_color_countPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(
              ffi.Int64, ffi.Pointer<wire_ImageData>)>>('wire_color_count');
  late final _wire_color_count = _wire_color_countPtr
      .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_groundline_sample_overlay(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_groundline_sample_overlay(
      port_,
      img,
    );
  }

  late final _wire_groundline_sample_overlayPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_ImageData>)>>('wire_groundline_sample_overlay');
  late final _wire_groundline_sample_overlay =
      _wire_groundline_sample_overlayPtr
          .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_start_rgb_kmeans_training(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_start_rgb_kmeans_training(
      port_,
      img,
    );
  }

  late final _wire_start_rgb_kmeans_trainingPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_ImageData>)>>('wire_start_rgb_kmeans_training');
  late final _wire_start_rgb_kmeans_training =
      _wire_start_rgb_kmeans_trainingPtr
          .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_start_hs_kmeans_training(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_start_hs_kmeans_training(
      port_,
      img,
    );
  }

  late final _wire_start_hs_kmeans_trainingPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_ImageData>)>>('wire_start_hs_kmeans_training');
  late final _wire_start_hs_kmeans_training = _wire_start_hs_kmeans_trainingPtr
      .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_color_clusterer(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_color_clusterer(
      port_,
      img,
    );
  }

  late final _wire_color_clustererPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(
              ffi.Int64, ffi.Pointer<wire_ImageData>)>>('wire_color_clusterer');
  late final _wire_color_clusterer = _wire_color_clustererPtr
      .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_groundline_overlay_k_means(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_groundline_overlay_k_means(
      port_,
      img,
    );
  }

  late final _wire_groundline_overlay_k_meansPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_ImageData>)>>('wire_groundline_overlay_k_means');
  late final _wire_groundline_overlay_k_means =
      _wire_groundline_overlay_k_meansPtr
          .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_groundline_filter_k_means(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_groundline_filter_k_means(
      port_,
      img,
    );
  }

  late final _wire_groundline_filter_k_meansPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_ImageData>)>>('wire_groundline_filter_k_means');
  late final _wire_groundline_filter_k_means =
      _wire_groundline_filter_k_meansPtr
          .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_hs_groundline_filter_k_means(
    int port_,
    ffi.Pointer<wire_ImageData> img,
  ) {
    return _wire_hs_groundline_filter_k_means(
      port_,
      img,
    );
  }

  late final _wire_hs_groundline_filter_k_meansPtr = _lookup<
          ffi.NativeFunction<
              ffi.Void Function(ffi.Int64, ffi.Pointer<wire_ImageData>)>>(
      'wire_hs_groundline_filter_k_means');
  late final _wire_hs_groundline_filter_k_means =
      _wire_hs_groundline_filter_k_meansPtr
          .asFunction<void Function(int, ffi.Pointer<wire_ImageData>)>();

  void wire_get_correlation_flow(
    int port_,
    ffi.Pointer<wire_uint_8_list> prev_ys,
    ffi.Pointer<wire_uint_8_list> current_ys,
    int width,
    int height,
  ) {
    return _wire_get_correlation_flow(
      port_,
      prev_ys,
      current_ys,
      width,
      height,
    );
  }

  late final _wire_get_correlation_flowPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(
              ffi.Int64,
              ffi.Pointer<wire_uint_8_list>,
              ffi.Pointer<wire_uint_8_list>,
              ffi.Int64,
              ffi.Int64)>>('wire_get_correlation_flow');
  late final _wire_get_correlation_flow =
      _wire_get_correlation_flowPtr.asFunction<
          void Function(int, ffi.Pointer<wire_uint_8_list>,
              ffi.Pointer<wire_uint_8_list>, int, int)>();

  void wire_reset_position_estimate(
    int port_,
  ) {
    return _wire_reset_position_estimate(
      port_,
    );
  }

  late final _wire_reset_position_estimatePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(ffi.Int64)>>(
          'wire_reset_position_estimate');
  late final _wire_reset_position_estimate =
      _wire_reset_position_estimatePtr.asFunction<void Function(int)>();

  void wire_process_sensor_data(
    int port_,
    ffi.Pointer<wire_uint_8_list> incoming_data,
  ) {
    return _wire_process_sensor_data(
      port_,
      incoming_data,
    );
  }

  late final _wire_process_sensor_dataPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_uint_8_list>)>>('wire_process_sensor_data');
  late final _wire_process_sensor_data = _wire_process_sensor_dataPtr
      .asFunction<void Function(int, ffi.Pointer<wire_uint_8_list>)>();

  void wire_parse_sensor_data(
    int port_,
    ffi.Pointer<wire_uint_8_list> incoming_data,
  ) {
    return _wire_parse_sensor_data(
      port_,
      incoming_data,
    );
  }

  late final _wire_parse_sensor_dataPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(ffi.Int64,
              ffi.Pointer<wire_uint_8_list>)>>('wire_parse_sensor_data');
  late final _wire_parse_sensor_data = _wire_parse_sensor_dataPtr
      .asFunction<void Function(int, ffi.Pointer<wire_uint_8_list>)>();

  ffi.Pointer<wire_ImageData> new_box_autoadd_image_data_0() {
    return _new_box_autoadd_image_data_0();
  }

  late final _new_box_autoadd_image_data_0Ptr =
      _lookup<ffi.NativeFunction<ffi.Pointer<wire_ImageData> Function()>>(
          'new_box_autoadd_image_data_0');
  late final _new_box_autoadd_image_data_0 = _new_box_autoadd_image_data_0Ptr
      .asFunction<ffi.Pointer<wire_ImageData> Function()>();

  ffi.Pointer<wire_uint_8_list> new_uint_8_list_0(
    int len,
  ) {
    return _new_uint_8_list_0(
      len,
    );
  }

  late final _new_uint_8_list_0Ptr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<wire_uint_8_list> Function(
              ffi.Int32)>>('new_uint_8_list_0');
  late final _new_uint_8_list_0 = _new_uint_8_list_0Ptr
      .asFunction<ffi.Pointer<wire_uint_8_list> Function(int)>();

  void free_WireSyncReturnStruct(
    WireSyncReturnStruct val,
  ) {
    return _free_WireSyncReturnStruct(
      val,
    );
  }

  late final _free_WireSyncReturnStructPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(WireSyncReturnStruct)>>(
          'free_WireSyncReturnStruct');
  late final _free_WireSyncReturnStruct = _free_WireSyncReturnStructPtr
      .asFunction<void Function(WireSyncReturnStruct)>();
}

class _Dart_Handle extends ffi.Opaque {}

class wire_uint_8_list extends ffi.Struct {
  external ffi.Pointer<ffi.Uint8> ptr;

  @ffi.Int32()
  external int len;
}

class wire_ImageData extends ffi.Struct {
  external ffi.Pointer<wire_uint_8_list> ys;

  external ffi.Pointer<wire_uint_8_list> us;

  external ffi.Pointer<wire_uint_8_list> vs;

  @ffi.Int64()
  external int width;

  @ffi.Int64()
  external int height;

  @ffi.Int64()
  external int uv_row_stride;

  @ffi.Int64()
  external int uv_pixel_stride;
}

typedef DartPostCObjectFnType = ffi.Pointer<
    ffi.NativeFunction<ffi.Bool Function(DartPort, ffi.Pointer<ffi.Void>)>>;
typedef DartPort = ffi.Int64;
typedef uintptr_t = ffi.UnsignedLongLong;

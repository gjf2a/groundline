import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'flutter_vision.dart';
import 'main.dart';

class SimpleImageRunner extends VisionRunner {
  final CameraImagePainter _livePicture = CameraImagePainter(makeColorFrom);

  @override
  CameraImagePainter livePicture() {
    return _livePicture;
  }

  @override
  Widget display(SelectorPageState selector) {
    return MaterialApp(
        home: Scaffold(
            appBar: AppBar(
                title: const Text("This is a title")),
            body: Center(
                child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      CustomPaint(painter: _livePicture),
                      Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          selector.startStopButton(),
                          Text(selector.ipAddr),
                          Text("Grabbed: ${_livePicture.frameCount()} (${_livePicture.width()} x ${_livePicture.height()}) FPS: ${_livePicture.fps().toStringAsFixed(2)}"),
                          Text(selector.incoming),
                          selector.returnToStartButton(),
                        ],
                      ),
                    ]
                )
            )
        )
    );
  }
}

class KMeansImageRunner extends VisionRunner {
  final KMeansGroundlineTrainer _livePicture = KMeansGroundlineTrainer();

  @override
  CameraImagePainter livePicture() {
    return _livePicture;
  }

  @override
  Widget display(SelectorPageState selector) {
    return MaterialApp(
        home: Scaffold(
            appBar: AppBar(
                title: const Text("This is a title")),
            body: Center(
                child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      CustomPaint(painter: _livePicture),
                      Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          selector.startStopButton(),
                          Text(selector.ipAddr),
                          Text("Grabbed: ${_livePicture.frameCount()} (${_livePicture.width()} x ${_livePicture.height()}) FPS: ${_livePicture.fps().toStringAsFixed(2)}"),
                          Text(selector.incoming),
                          _trainKmeansButton(),
                          selector.returnToStartButton(),
                        ],
                      ),
                    ]
                )
            )
        )
    );
  }

  Widget _trainKmeansButton() {
    return makeCmdButton("Train k-means", Colors.green, () {
      _livePicture.startTraining = true;
    });
  }
}



class GroundlineCounter extends CameraImagePainter {
  int colorCount = 0;
  GroundlineCounter() : super(makeGroundlineSampleOverlay);

  @override
  Future<void> setImage(CameraImage img) async {
    super.setImage(img);
    colorCount = await api.colorCount(img: from(img));
  }
}

class KMeansGroundlineTrainer extends CameraImagePainter {
  KMeansGroundlineTrainer() : super(makeGroundlineKmeansFilter);
  bool startTraining = false;

  @override
  Future<void> setImage(CameraImage img) async {
    super.setImage(img);
    if (startTraining) {
      await api.startKmeansTraining(img: from(img));
      startTraining = false;
      resetFps();
    }
  }
}